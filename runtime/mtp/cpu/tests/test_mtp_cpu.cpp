#include "../mtp_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using namespace MLIPCore;
using namespace MLIPCore::Cpu;

void close(double actual, double expected, double tolerance, const char *what) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance*(1.0+std::abs(expected)))
        throw std::runtime_error(std::string(what)+": "+std::to_string(actual)+" != "+std::to_string(expected));
}

struct Fixture {
    mlip model{};
    std::array<alpha_basic_t,4> basic{{{0,0,0,0},{1,1,0,0},{1,0,1,0},{1,0,0,1}}};
    std::array<alpha_times_t,5> products{{{1,1,1,4},{2,2,1,4},{3,3,1,4},{0,4,1,5},{0,0,1,6}}};
    std::array<int,5> scalar{{0,4,5,6,4}}; // Deliberately duplicate scalar 4.
    std::array<mlip::radial_basis_t,4> radial;
    std::vector<double> radial_coefficients, species, projection, coefficients;
    explicit Fixture(mlip::site_model_type head, mlip::activation_type activation) {
        model.site_model=head; model.model_activation=activation;
        model.species_count=2; model.descriptor_count=head==mlip::MODEL_MTPM_LI ? 1 :
            (head==mlip::MODEL_MTPM_Q2 || head==mlip::MODEL_MTPM_FS ? 2 : 3);
        model.model_hidden1=(head==mlip::MODEL_MTPM_NN1 || head==mlip::MODEL_MTPM_NN2) ? 4 : 0;
        model.model_hidden2=head==mlip::MODEL_MTPM_NN2 ? 3 : 0;
        model.alpha_index_basic_count=4; model.alpha_moments_count=7;
        model.alpha_index_times_count=5; model.alpha_scalar_moments=5;
        model.alpha_index_basic=basic.data(); model.alpha_index_times=products.data();
        model.alpha_moment_mapping=scalar.data(); model.max_alpha_index_basic=2;
        model.max_mu=2; model.radial_func_count=2; model.rb_size=3;
        model.cutoff=4.3; model.min_dist=0.4; model.max_dist=4.3; model.cutoffsq=4.3*4.3;
        model.pair_radial_basis_count=4; model.pair_radial_bases=radial.data();
        for(int pair=0;pair<4;++pair) {
            radial[pair].rb=mlip::RB_CHEBYCONFIGURABLE;
            radial[pair].cutoff=mlip::RB_CUTOFF_NORMALIZED_QUADRATIC;
            radial[pair].transform=mlip::RB_TRANSFORM_LINEAR;
            radial[pair].size=3; radial[pair].min_dist=0.4;
            radial[pair].max_dist=4.0+0.1*pair;
            radial[pair].scaling=0.7+0.1*pair;
        }
        radial_coefficients.resize(4*2*3);
        for(std::size_t i=0;i<radial_coefficients.size();++i)
            radial_coefficients[i]=0.1+0.013*static_cast<double>(i%7);
        model.radial_coeffs=radial_coefficients.data();
        species.resize(2*(model.descriptor_count+1));
        for(std::size_t i=0;i<species.size();++i) species[i]=0.01*(i+1);
        model.species_coeffs=species.data(); model.descriptor_species_coeffs=species.data()+2;
        projection.resize(5*model.descriptor_count);
        for(std::size_t i=0;i<projection.size();++i) projection[i]=0.03*(static_cast<int>(i%5)-2);
        model.moment_coeffs=projection.data(); model.descriptor_moment_coeffs=projection.data();
        const int q=model.descriptor_count,h1=model.model_hidden1,h2=model.model_hidden2;
        const int count=head==mlip::MODEL_MTPM_Q2 ? 5 : (head==mlip::MODEL_MTPM_FS ? 1 :
            (head==mlip::MODEL_MTPM_NN1 ? q*h1+2*h1 :
             (head==mlip::MODEL_MTPM_NN2 ? q*h1+h1+h1*h2+2*h2 : 0)));
        coefficients.resize(count);
        for(int i=0;i<count;++i) coefficients[i]=0.05+0.003*(i%11);
        model.model_coeffs=coefficients.data(); model.model_coeffs_count=count;
    }
};

using Coordinates=std::array<std::array<double,3>,5>;
struct Geometry {
    Coordinates xyz{{{0.2,0.3,0.4},{1.1,-0.2,0.7},{-0.5,1.3,0.2},{0.4,0.1,1.8},{1.2,1.1,0.9}}};
    std::array<const double*,5> positions;
    std::array<int,5> types{{1,2,1,2,2}};
    std::array<int,3> mapping{{-1,0,1}};
    std::array<int,4> neighbors{{1 | (1<<30),2,3,4}};
    Geometry() { for(int i=0;i<5;++i) positions[i]=xyz[i].data(); }
    double eval(const Executor &executor,Workspace &workspace,bool energy=true,bool derivatives=true,int n=4) {
        return executor.evaluate_site(0,n,neighbors.data(),types.data(),positions.data(),
                                      mapping.data(),energy,derivatives,workspace);
    }
};

void test_head(mlip::site_model_type head,mlip::activation_type activation) {
    Fixture fixture(head,activation);
    Executor executor(fixture.model);
    Workspace cached(fixture.model),recomputed(fixture.model);
    cached.reserve(fixture.model,4,DerivativeMode::Cached);
    recomputed.reserve(fixture.model,4,DerivativeMode::Recompute);
    Geometry geometry;
    const double energy=geometry.eval(executor,cached);
    close(geometry.eval(executor,recomputed),energy,1e-14,"strategy energy");
    std::array<std::array<double,3>,4> gradients;
    for(int edge=0;edge<4;++edge) for(int a=0;a<3;++a) {
        gradients[edge][a]=cached.gradient(edge)[a];
        close(recomputed.gradient(edge)[a],gradients[edge][a],1e-13,"strategy gradient");
    }
    close(geometry.eval(executor,cached,true,false),energy,1e-14,"energy only");
    close(geometry.eval(executor,cached,false,true),0.0,0.0,"force-only energy output");
    for(int edge=0;edge<4;++edge) for(int a=0;a<3;++a)
        close(cached.gradient(edge)[a],gradients[edge][a],1e-14,"force-only gradient");
    const auto bytes=cached.allocated_bytes();
    cached.reserve(fixture.model,4,DerivativeMode::Cached);
    if(bytes!=cached.allocated_bytes()) throw std::runtime_error("Unnecessary workspace growth");
    constexpr double step=2e-6;
    for(int atom=0;atom<5;++atom) for(int a=0;a<3;++a) {
        const double saved=geometry.xyz[atom][a];
        geometry.xyz[atom][a]=saved+step;
        const double plus=geometry.eval(executor,cached,true,false);
        geometry.xyz[atom][a]=saved-step;
        const double minus=geometry.eval(executor,cached,true,false);
        geometry.xyz[atom][a]=saved;
        double analytic=atom ? gradients[atom-1][a] : 0.0;
        if(!atom) for(int edge=0;edge<4;++edge) analytic-=gradients[edge][a];
        close((plus-minus)/(2*step),analytic,2e-8,"coordinate derivative");
    }
    const std::array<std::array<int,2>,6> components{{{0,0},{1,1},{2,2},{0,1},{0,2},{1,2}}};
    const Coordinates original=geometry.xyz;
    for(auto component:components) {
        const int a=component[0],b=component[1];
        double derivative=0.0;
        for(int edge=0;edge<4;++edge) {
            const auto &r=original[edge+1]; const auto &x=original[0];
            derivative+=0.5*((r[a]-x[a])*gradients[edge][b]+(r[b]-x[b])*gradients[edge][a]);
        }
        auto deformed=[&](double strain) {
            geometry.xyz=original;
            for(int i=0;i<5;++i) {
                geometry.xyz[i][a]+=0.5*strain*original[i][b];
                geometry.xyz[i][b]+=0.5*strain*original[i][a];
            }
            return geometry.eval(executor,cached,true,false);
        };
        const double plus=deformed(step),minus=deformed(-step);
        close((plus-minus)/(2*step),derivative,2e-8,"strain derivative");
    }
    geometry.xyz=original;
    std::reverse(geometry.neighbors.begin(),geometry.neighbors.end());
    close(geometry.eval(executor,cached),energy,1e-13,"neighbor permutation energy");
    for(int edge=0;edge<4;++edge) for(int a=0;a<3;++a)
        close(cached.gradient(edge)[a],gradients[3-edge][a],1e-13,"neighbor permutation force");
    const double isolated=geometry.eval(executor,cached,true,true,0);
    close(geometry.eval(executor,recomputed,true,true,0),isolated,1e-14,"isolated center");
    if(cached.edge_count()) throw std::runtime_error("Stale edge count for isolated center");
    const auto cached_bytes=cached.allocated_bytes();
    cached.reserve(fixture.model,4,DerivativeMode::Recompute);
    if(cached.allocated_bytes()>=cached_bytes) throw std::runtime_error("Recompute did not release Jacobian");
#ifdef _OPENMP
    // The same immutable plan/model is intentionally shared by all workers.
    std::array<std::unique_ptr<Workspace>,4> workers;
    std::array<double,4> energies;
    for(auto &worker:workers) {
        worker=std::make_unique<Workspace>(fixture.model);
        worker->reserve(fixture.model,4,DerivativeMode::Cached);
    }
    #pragma omp parallel for num_threads(4)
    for(int thread=0;thread<4;++thread) {
        Geometry local;
        energies[thread]=local.eval(executor,*workers[thread]);
    }
    for(int thread=0;thread<4;++thread) {
        close(energies[thread],energy,1e-14,"thread-local energy");
        for(int edge=0;edge<4;++edge) for(int a=0;a<3;++a)
            close(workers[thread]->gradient(edge)[a],gradients[edge][a],1e-14,"thread-local gradient");
    }
#endif
}
}

int main() {
    try {
        for(auto head:{mlip::MODEL_MTPM_LI,mlip::MODEL_MTPM_Q2,mlip::MODEL_MTPM_FS,
                       mlip::MODEL_MTPM_NN1,mlip::MODEL_MTPM_NN2})
            for(auto activation:{mlip::ACTIVATION_TANH,mlip::ACTIVATION_RELU,
                                 mlip::ACTIVATION_SIGMOID,mlip::ACTIVATION_LINEAR}) {
                if(head==mlip::MODEL_MTPM_NN2 && activation==mlip::ACTIVATION_LINEAR) continue;
                test_head(head,activation);
            }
        std::cout << "MTP_M CPU: all heads, activations, duplicate scalars, cached/recompute, flags and finite differences passed\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
