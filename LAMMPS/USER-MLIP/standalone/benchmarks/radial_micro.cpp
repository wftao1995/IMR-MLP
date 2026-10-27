// Isolate host radial arithmetic from the descriptor graph and LAMMPS overhead.
// Override MTP_RADIAL_HEADER with a frozen earlier header for a paired build.
#ifndef MTP_RADIAL_HEADER
#define MTP_RADIAL_HEADER "radial_basis.h"
#endif
#include MTP_RADIAL_HEADER
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

using Basis = MLIPCore::mlip::radial_basis_t;
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void evaluate(const Basis &b, double r, double *v, double *d) {
    MLIPCore::Radial::evaluate(b,r,v,d);
}

int main(int argc,char **argv) {
    if(argc!=4) {
        std::cerr<<"Usage: radial_micro TYPE SIZE ITERATIONS\n"
                   "TYPE: chebyshev, cosine, poly4, poly5, configurable, flipped, polyb, polybcos\n";
        return 2;
    }
    try {
        const std::string type=argv[1];
        Basis b;
        b.size=std::stoi(argv[2]);
        const auto iterations=std::stoll(argv[3]);
        if(b.size<1 || b.size>128 || iterations<1) throw std::invalid_argument("invalid size/iterations");
        std::string name,extra;
        if(type=="chebyshev") name="RBChebyshev";
        else if(type=="cosine") name="RBChebyCos";
        else if(type=="poly4") {name="RBChebyshevPoly4";extra="rb_b = -0.7\n";}
        else if(type=="poly5") {name="RBChebyshevPoly5";extra="rb_b = -0.7\n";}
        else if(type=="polyb") name="RBPolyB";
        else if(type=="polybcos") name="RBPolyBCos";
        else if(type=="configurable" || type=="flipped") {
            name="RBChebyshevConfigurable";
            extra="basis_function_type = "+std::string(type=="flipped"?"one_minus_chebyshev":"chebyshev")+
                  "\ncutoff_type = poly5\ncutoff_b = -0.7\nshift_type = exponential\nshift_lambda = -0.7\n";
        } else throw std::invalid_argument("unknown type");
        // Use real initialization, including any constants prepared by this version.
        std::istringstream input("= "+name+"\nmin_dist = 0.5\nmax_dist = 5\nradial_basis_size = "+
                                 std::to_string(b.size)+"\n"+extra+"radial_funcs_count");
        b=MLIPCore::Radial::read_radial_basis(input,.65).basis;
        // Synthetic dense matrices exercise the contraction without initialization.
        // Numerical fidelity uses the independent native oracle, not these data.
        if(MLIPCore::Radial::is_poly(b)) {
            b.polynomial_matrix.resize(b.size*b.size);
            b.polynomial_normalizations.resize(b.size);
            for(int j=0;j<b.size;++j) {
                b.polynomial_normalizations[j]=std::pow(.2,j+3);
                for(int i=0;i<b.size;++i)b.polynomial_matrix[j*b.size+i]=std::sin(i+2*j+.3);
            }
        }
        std::vector<double> v(b.size),d(b.size),distances(4096);
        for(int i=0;i<4096;++i)distances[i]=.7+4.2*((i*173)%4096)/4096.;
        double checksum=0;
        for(int i=0;i<100000;++i) {
            evaluate(b,distances[i%4096],v.data(),d.data());
            checksum+=v[0]+d[0];
        }
        const auto start=std::chrono::steady_clock::now();
        for(long long i=0;i<iterations;++i) {
            evaluate(b,distances[i%4096],v.data(),d.data());
            checksum+=v[i%b.size]+d[i%b.size];
        }
        const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/iterations;
        std::cout<<std::setprecision(17)<<"{\"ns_per_evaluation\":"<<ns<<",\"checksum\":"<<checksum<<"}\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
