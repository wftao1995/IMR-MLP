#include "potentials/model_bundle.h"
#include "potentials/potential_wrapper.h"
#include "active_learn/al_potential.h"
#include "utils/mpi_stubs.h"
#include <iostream>

namespace {
template<class F> int on_writer(F action) {
    std::string error;
    int rank=0;
#ifdef IMRMLP_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
    if(rank==0)try {action();}catch(const std::exception&e){error=e.what();}
#ifdef IMRMLP_MPI
    int size=error.size();MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);error.resize(size);
    if(size)MPI_Bcast(error.data(),size,MPI_CHAR,0,MPI_COMM_WORLD);
#endif
    if(!error.empty())throw std::runtime_error(error);
    return 0;
}
}

int model_convert(int argc,char **argv) {
    std::string input,output,settings_file,type="auto",index,profile="preserve";
    auto &options=CMDOption::global_cmdopts;
    options.addop(input,"input","in","Legacy model or GPM input")
        .addop(output,"output","out","Output GPM file",true)
        .addop(settings_file,"settings","","Existing potential and optional al: configuration")
        .addop(type,"type","","Legacy potential type; GPM infers its type")
        .addop(index,"index","","Explicit ordered element map")
        .addop(profile,"profile","","preserve, inference, or resume")
        .phrase_options(argc,argv);
    if(options.displayhelp){options.showhelp(std::cout);return 0;}
    return on_writer([&]{
        GPSettings all;if(!settings_file.empty())all=GPSettings(settings_file);
        auto settings=all.ExtractSubSettings("pot");
        if(!input.empty())settings.Modify("potfile",input);
        if(type!="auto"||settings["pottype"].empty())settings.Modify("pottype",type);
        if(!index.empty())settings.Modify("typemap",index);
        auto bundle=Potentials::ModelBundle::from_settings(settings);
        auto al_settings=all.ExtractSubSettings("al");
        if(profile!="inference")if(auto al=AL::ALPotential::create(al_settings)) {
            al->init_builder(bundle->potential());bundle->attach_al(*al);
        }
        using Profile=Potentials::ModelBundle::Profile;
        GPM::require(profile=="preserve"||profile=="inference"||profile=="resume","unknown save profile");
        bundle->save(output,profile=="inference"?Profile::Inference:profile=="resume"?Profile::Resume:Profile::Preserve);
        std::cout<<"Saved "<<output<<'\n';
    });
}

int model_info(int argc,char **argv) {
    std::string input;auto &options=CMDOption::global_cmdopts;
    options.addop(input,"input","in","GPM model",true).phrase_options(argc,argv);
    if(options.displayhelp){options.showhelp(std::cout);return 0;}
    return on_writer([&]{std::cout<<GPM::Value(Potentials::ModelBundle::inspect(input)).serialize(true)<<'\n';});
}

int model_validate(int argc,char **argv) {
    std::string input;auto &options=CMDOption::global_cmdopts;
    options.addop(input,"input","in","GPM model",true).phrase_options(argc,argv);
    if(options.displayhelp){options.showhelp(std::cout);return 0;}
    return on_writer([&]{Potentials::ModelBundle::validate(input);std::cout<<"Valid GPM: "<<input<<'\n';});
}
