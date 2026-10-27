#pragma once
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace StandaloneRadialFixture {
inline const std::vector<std::string> cutoffs={"quadratic","cosine","poly5","normalized_quadratic",
    "cosine_left","cosine_unscaled","cosine_squared","sine_squared","poly3","poly4","smoothstep5","smoothstep7"};
inline const std::vector<std::string> transforms={"linear","exponential","exponential_left","i2","exponential_tanh"};
inline std::string block(const std::string &name,int size=4,int cut=0,int shift=0,bool flip=false,
                         double lo=.5,double hi=5,double lambda=.7,double scale=.8) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"radial_basis_type = "<<name<<"\nmin_dist = "<<lo<<"\nmax_dist = "<<hi<<"\nradial_basis_size = "<<size<<'\n';
    if(name=="RBChebyshevPoly4" || name=="RBChebyshevPoly5") out<<"rb_b = -0.7\n";
    if(name=="RBChebyshevConfigurable") {
        out<<"basis_function_type = "<<(flip?"one_minus_chebyshev":"chebyshev")<<"\ncutoff_type = "<<cutoffs[cut]<<'\n';
        if(cut==2 || cut==8 || cut==9) out<<"cutoff_b = -0.7\n";
        if(cut==11)out<<"cutoff_r1 = "<<lo+.2*(hi-lo)<<"\ncutoff_r2 = "<<lo+.7*(hi-lo)<<'\n';
        out<<"shift_type = "<<transforms[shift]<<'\n';
        if(shift!=0 && shift!=3)out<<"shift_lambda = "<<lambda<<'\n';
        if(shift==4)out<<"shift_scale_lambda = "<<scale<<'\n';
    }
    return out.str();
}
inline std::string model(const std::vector<std::string> &blocks,int size=4,const std::string &head="classic") {
    const int species=2,q=head=="li"?1:2,h=3,h2=2;
    const bool neural=head=="n1" || head=="n2";
    const int count=head=="q2"?5:head=="fs"?1:head=="n1"?q*h+2*h:head=="n2"?q*h+h+h*h2+2*h2:0;
    std::ostringstream out;out<<std::setprecision(17);
    out<<"MTP\nversion = 1.1.0\npotential_name = standalone_radial\nscaling = 0.65\nspecies_count = 2\npotential_tag = ";
    if(head!="classic") {
        out<<"{\"schema\":\"gpml.mtp.metadata\",\"version\":1,\"type\":\"MTP_M\",\"model\":\"mtp_m_"<<head
           <<"\",\"q\":"<<q<<",\"model_coeffs_count\":"<<count;
        if(neural)out<<",\"nn\":{\"activation\":\"TANH\",\"hidden_layers\":[3"<<(head=="n2"?",2":"")<<"]}";
        out<<"}";
    }
    out<<'\n';
    if(blocks.size()==1)out<<blocks[0];
    else {
        out<<"radial_basis_split = 1\n";
        // Deliberately permute file order.
        for(int pair:{3,1,0,2})out<<"pair "<<pair/2<<' '<<pair%2<<'\n'<<blocks[pair];
    }
    out<<"radial_funcs_count = 1\nradial_coeffs\n";
    for(int i=0;i<species*species;++i) {
        out<<i/2<<'-'<<i%2<<"\n{";
        for(int j=0;j<size;++j)out<<(j?",":"")<<.025*(1+.1*i)/(j+1)*(j%2?-1:1);
        out<<"}\n";
    }
    out<<"alpha_moments_count = 5\nalpha_index_basic_count = 4\n"
       <<"alpha_index_basic = {{0,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}\n"
       <<"alpha_index_times_count = 3\nalpha_index_times = {{1,1,1,4},{2,2,1,4},{3,3,1,4}}\n"
       <<"alpha_scalar_moments = 2\nalpha_moment_mapping = {0,4}\nspecies_coeffs = {0.02,-0.01";
    if(head!="classic")for(int i=0;i<species*q;++i)out<<','<<.01*(i+1);
    out<<"}\nmoment_coeffs = {";
    for(int i=0;i<(head=="classic"?2:2*q);++i)out<<(i?",":"")<<.03/(i+1);
    out<<"}\n";
    if(head!="classic") {
        out<<"model_coeffs = {";
        for(int i=0;i<count;++i)out<<(i?",":"")<<.02*(i%3-1);
        out<<"}\n";
    }
    return out.str();
}
}
