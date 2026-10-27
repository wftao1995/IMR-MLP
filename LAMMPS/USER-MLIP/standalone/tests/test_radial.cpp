#include "radial_fixture.h"
#include "cpu/mtp_cpu.h"
#include "radial_cache.h"
#include <iostream>
#include <array>

namespace R=MLIPCore::Radial;
namespace F=StandaloneRadialFixture;
using namespace MLIPCore::Cpu;
void close(double a,double b,double tolerance) {
    if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>tolerance*(1+std::abs(b)))
        throw std::runtime_error("standalone radial numerical mismatch: "+std::to_string(a)+" != "+std::to_string(b));
}
Model load(const std::string &text) {
    int map[]={-1,0,1};double cutoff;
    return Model(MLIPCore::setup_from_text(text,map,&cutoff,3));
}
void verify(const std::vector<std::string> &blocks,int size) {
    for(const std::string head:{"classic","li","q2","fs","n1","n2"}) {
        const auto text=F::model(blocks,size,head);auto model=load(text);
        std::vector<MLIPCore::mlip::radial_basis_t *> pointers;
        // Freeze a distinctive matrix, proving that the footer takes priority over initialization.
        if(model->pair_radial_basis_count) {
            for(int i=0;i<(blocks.size()==1?1:4);++i) {
                auto &b=model->pair_radial_bases[i];
                if(R::is_poly(b))for(auto &x:b.polynomial_matrix)x*=1.01;
                pointers.push_back(&b);
            }
            auto deployed=load(text+R::cache_footer(pointers,blocks.size()!=1,2));
            close(deployed->pair_radial_bases[0].polynomial_matrix.empty()?0:deployed->pair_radial_bases[0].polynomial_matrix[0],
                  pointers[0]->polynomial_matrix.empty()?0:pointers[0]->polynomial_matrix[0],0);
            model=std::move(deployed);
        }
        Executor executor(*model);Workspace cached(*model),recomputed(*model);
        cached.reserve(*model,4,DerivativeMode::Cached);recomputed.reserve(*model,4,DerivativeMode::Recompute);
        std::array<std::array<double,3>,5> xyz{{{{0,0,0}},{{.3,.1,0}},{{1.1,.3,.2}},{{3.7,.2,.1}},{{.2,2.1,1.4}}}};
        const double *positions[5];for(int i=0;i<5;++i)positions[i]=xyz[i].data();
        int types[]={1,2,1,2,1},mapping[]={-1,0,1},neighbors[]={1,2,3,4};
        auto energy=[&](Workspace &w,bool derivatives) {return executor.evaluate_site(0,4,neighbors,types,positions,mapping,true,derivatives,w);};
        close(energy(cached,true),energy(recomputed,true),1e-12);
        std::array<std::array<double,3>,4> gradients{};
        for(int e=0;e<cached.edge_count();++e)for(int a=0;a<3;++a) {
            const int atom=cached.neighbor(e);gradients[atom-1][a]=cached.gradient(e)[a];
            close(cached.gradient(e)[a],recomputed.gradient(e)[a],1e-12);
        }
        const double step=2e-6;
        for(int atom=1;atom<5;++atom)for(int a=0;a<3;++a) {
            const auto saved=xyz[atom][a];
            xyz[atom][a]=saved+step;const auto plus=energy(cached,false);
            xyz[atom][a]=saved-step;const auto minus=energy(cached,false);xyz[atom][a]=saved;
            close((plus-minus)/(2*step),gradients[atom-1][a],2e-7);
        }
        for(int a=0;a<3;++a)for(int b=a;b<3;++b) {
            const auto original=xyz;double derivative=0;
            for(int i=1;i<5;++i)derivative+=.5*(gradients[i-1][a]*xyz[i][b]+gradients[i-1][b]*xyz[i][a]);
            auto strain=[&](double h) {
                xyz=original;
                for(int i=1;i<5;++i) {xyz[i][a]+=.5*h*original[i][b];xyz[i][b]+=.5*h*original[i][a];}
                return energy(cached,false);
            };
            const double plus=strain(step),minus=strain(-step);xyz=original;
            close((plus-minus)/(2*step),derivative,2e-7);
        }
    }
}
int main() {
    try {
        for(int size:{1,4}) {
            for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos"})verify({F::block(name,size)},size);
            for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true})
                verify({F::block("RBChebyshevConfigurable",size,cut,shift,flip)},size);
        }
        verify({F::block("RBPolyB",4,0,0,false,.4,4.1),F::block("RBPolyBCos",4,0,0,false,.6,4.4),
            F::block("RBChebyshevConfigurable",4,11,4,true,.8,4.7),F::block("RBChebyshevPoly4")},4);
        for(int size:{2,6,8})for(const auto *name:{"RBPolyB","RBPolyBCos"})verify({F::block(name,size)},size);
        std::cout<<"Standalone: six bases, 120 configurable combinations, six heads, split, single basis, fixed matrices and coordinate/strain differences passed\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
