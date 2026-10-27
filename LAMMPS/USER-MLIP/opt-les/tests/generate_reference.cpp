// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "potentials/potential_mtp_LES.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
using namespace Potentials;
int main(int argc,char** argv) {
    if(argc!=3 && argc!=4)return 2;
    const int channels=argc==4?std::stoi(argv[3]):1;
    if(channels<1 || channels>16)return 2;
    const std::filesystem::path out=argv[2];std::filesystem::create_directories(out);
    std::ofstream manifest(out/"cases.txt");
    for(int geometry=0;geometry<4;++geometry)for(int mode=0;mode<5;++mode) {
        PotMTP_LES p(argv[1]);p.set_element_map({{0,"Al"},{1,"Ni"}});
        p.les_species_coeffs[0]=.23;p.les_species_coeffs[1]=-.17;
        std::fill_n(p.les_moment_coeffs,p.get_alpha_scalar_moments(),0.);
        p.les_moment_coeffs[0]=.35;p.les_moment_coeffs[1]=-.21;p.les_moment_coeffs[3]=.12;
        p.dl=2.0;
        if(mode<4){p.les_kernel="sog";p.sog_terms=2;p.sog_channels=channels;}
        p.train_init();
        if(mode<4){
            for(int c=0;c<channels;++c){
                p.les_species_coeffs[2*c]=.23+.1*c;p.les_species_coeffs[2*c+1]=-.17+.07*c;
                for(int a=0;a<p.get_alpha_scalar_moments();++a)p.les_moment_coeffs[c*p.get_alpha_scalar_moments()+a]=a==0?.35+.1*c:a==1?-.21:a==3?.12:0.;
                p.model_les.sog_weights[2*c]=4.2+c;p.model_les.sog_weights[2*c+1]=-1.3-.2*c;
                p.model_les.sog_log_bandwidths[2*c]=std::log(.7)+.1*c;p.model_les.sog_log_bandwidths[2*c+1]=std::log(1.2)-.1*c;
            }
            p.model_les.sog_include_zero=mode&1;p.remove_self_interaction=mode&2;p.train_init();
        }
        Atoms a;a.cell_<<10.,0,0,geometry?.8:0.,10.5,0,geometry?-.3:0.,geometry?.6:0.,11.;
        a.pos_={{.2,2.,2.},{9.2,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};a.chemmap_={0,1,0,1};
        if(geometry>=2){a.pos_.resize(1);a.chemmap_.resize(1);}
        if(geometry==3){a.cell_*=4.;a.pos_[0]={10.,10.,10.};}
        std::string name="g"+std::to_string(geometry)+"m"+std::to_string(mode);manifest<<name<<'\n';
        std::ofstream model(out/(name+".mtp"));p.Save(model);p.getEFS(a);
        std::ofstream reference(out/(name+".ref"));reference<<std::setprecision(17)<<a.energy_<<'\n';
        for(auto ij:{std::pair{0,0},{1,1},{2,2},{0,1},{0,2},{1,2}})reference<<a.stress_(ij.first,ij.second)<<' ';reference<<'\n';
        for(int i=0;i<a.size();++i){reference<<i+1<<' '<<a.forces_[i](0)<<' '<<a.forces_[i](1)<<' '<<a.forces_[i](2)<<' '<<a.site_energies_[i]<<' ';for(int c=0;c<p.channels();++c)reference<<p.LES_all_latents(i,c)<<' ';reference<<'\n';}
        std::ofstream data(out/(name+".data"));data<<std::setprecision(17)<<"SOG parity\n\n"<<a.size()<<" atoms\n2 atom types\n\n0 "<<a.cell_(0,0)<<" xlo xhi\n0 "<<a.cell_(1,1)<<" ylo yhi\n0 "<<a.cell_(2,2)<<" zlo zhi\n";
        data<<a.cell_(1,0)<<' '<<a.cell_(2,0)<<' '<<a.cell_(2,1)<<" xy xz yz\n\nAtoms # atomic\n\n";
        for(int i=0;i<a.size();++i)data<<i+1<<' '<<a.chemmap_[i]+1<<' '<<a.pos_[i]<<'\n';
    }
}
