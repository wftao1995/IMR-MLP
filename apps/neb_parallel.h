#pragma once
#include "mindriver/neb_evaluate.h"
#include "mindriver/neb.h"
#include "potentials/potential_wrapper.h"
#include <memory>
#include <iomanip>
#include <sstream>
#include "mindriver/neb_reporting.h"
#include <array>

// Rank zero owns the NEB driver. Workers service only explicit EFS requests,
// including endpoint, finite-difference and rejected-trial evaluations.
class NEBParallel {
    Potentials::AnyPotential *potential_ = nullptr;
    std::unique_ptr<Potentials::potential_wrapper> worker_potential_;
    std::string mode_ = "images";
    bool profiling_=false;
    int scope_=0;
    // Each scope: EFS, codec, collective/wait seconds, batches, image calls, batch wall.
    std::array<double,12> statistics_{};
    double *counter(int index) { return profiling_?&statistics_[6*scope_+index]:nullptr; }

#ifdef IMRMLP_MPI
    enum { stop, configure, evaluate, release_scope, collect_stats };
    void command(int value) { NEBReporting::Timer timer(counter(2)); MPI_Bcast(&value,1,MPI_INT,0,mpi.comm); }
    template<class F> void phase(F action, bool compute=false) {
        std::string error;
        int kind = 1;
        try { NEBReporting::Timer timer(counter(compute?0:1)); action(); }
        catch (const NEBInvalidGeometry &e) { error=e.what(); kind=2; }
        catch (const std::exception &e) { error=e.what(); }
        catch (...) { error="unknown calculator exception"; }
        NEBReporting::Timer timer(counter(2));
        int local=error.empty()?mpi.size:mpi.rank, first=mpi.size;
        MPI_Allreduce(&local,&first,1,MPI_INT,MPI_MIN,mpi.comm);
        if (first == mpi.size) return;
        MPI_Bcast(&kind,1,MPI_INT,first,mpi.comm);
        int length=error.size(); MPI_Bcast(&length,1,MPI_INT,first,mpi.comm);
        error.resize(length); MPI_Bcast(error.data(),length,MPI_CHAR,first,mpi.comm);
        error="NEB failure on rank "+std::to_string(first)+": "+error;
        if (kind==2) throw NEBInvalidGeometry(error);
        throw std::runtime_error(error);
    }
    void broadcast(std::string &data, int root) {
        unsigned long long size=data.size();
        { NEBReporting::Timer timer(counter(2)); MPI_Bcast(&size,1,MPI_UNSIGNED_LONG_LONG,root,mpi.comm); }
        phase([&] { data.resize(size); });
        NEBReporting::Timer timer(counter(2));
        for (size_t offset=0; offset<data.size();) {
            const int count=std::min<size_t>(65536,data.size()-offset);
            MPI_Bcast(data.data()+offset,count,MPI_CHAR,root,mpi.comm); offset+=count;
        }
    }
    void configure_all(const GPSettings *settings, bool profile=false) {
        std::string data;
        int configuration[2]={get_omp_max_threads(),int(profile)};
        MPI_Bcast(configuration,2,MPI_INT,0,mpi.comm);
        const int threads=configuration[0];
        phase([&] {
            int provided; MPI_Query_thread(&provided);
            if (provided<MPI_THREAD_FUNNELED && threads>1)
                throw std::runtime_error("NEB requires MPI_THREAD_FUNNELED for OpenMP");
            set_omp_num_threads(threads);
            if (mpi.rank==0) {
                std::ostringstream out;
                out<<std::quoted(mode_)<<'\n';
                for (const auto &[key,value]:settings->settings)
                    out<<std::quoted(key)<<' '<<std::quoted(value)<<'\n';
                data=out.str();
            }
        });
        broadcast(data,0);
        phase([&] {
            if (mpi.rank==0) return;
            std::istringstream in(data); in>>std::quoted(mode_);
            GPSettings settings; std::string key,value;
            while (in>>std::quoted(key)>>std::quoted(value)) settings.settings.emplace(key,value);
            worker_potential_=std::make_unique<Potentials::potential_wrapper>(settings);
            potential_=worker_potential_->pot;
        });
        profiling_=configuration[1];
    }
    void evaluate_all(std::vector<Atoms> &images, int begin, int end) {
        NEBReporting::Timer batch_timer(counter(5));
        if(profiling_) ++statistics_[scope_*6+3];
        int count=end-begin;
        { NEBReporting::Timer timer(counter(2)); MPI_Bcast(&count,1,MPI_INT,0,mpi.comm); }
        if(profiling_) statistics_[scope_*6+4]+=count>mpi.rank?(count-1-mpi.rank)/mpi.size+1:0;
        std::vector<Atoms> batch;
        phase([&] { batch.resize(count); });
        for (int i=0;i<count;++i) {
            std::string data;
            phase([&] {
                if (mpi.rank==0) {
                    std::ostringstream out(std::ios::binary);
                    images[begin+i].write_to_bin(out);
                    // The legacy in-memory binary codec predates PBC flags.
                    for (bool p:images[begin+i].pbc_) out.put(p?'1':'0');
                    if (!out) throw std::runtime_error("Cannot serialize NEB image");
                    data=out.str();
                }
            });
            broadcast(data,0);
            phase([&] {
                std::istringstream in(data,std::ios::binary);
                batch[i].read_from_bin(in);
                for (auto &p:batch[i].pbc_) p=in.get()=='1';
                if (!in) throw std::runtime_error("Cannot deserialize NEB image");
            });
        }
        phase([&] { evaluate_neb_images(*potential_,batch,0,count,mode_,mpi.rank,mpi.size); },true);
        // Fixed image order, without floating-point summation across ranks.
        for (int i=0;i<count;++i) {
            const int owner=i%mpi.size;
            std::string data;
            phase([&] {
                if (mpi.rank==owner) {
                    std::ostringstream out(std::ios::binary); batch[i].write_to_bin(out);
                    if (!out) throw std::runtime_error("Cannot serialize NEB prediction");
                    data=out.str();
                }
            });
            broadcast(data,owner);
            phase([&] {
                if (mpi.rank==0) {
                    std::istringstream in(data,std::ios::binary);
                    images[begin+i].read_from_bin(in);
                    if (!in) throw std::runtime_error("Cannot deserialize NEB prediction");
                }
            });
        }
    }
    std::vector<double> gather() {
        std::vector<double> all(mpi.rank==0?12*mpi.size:0);
        MPI_Gather(statistics_.data(),12,MPI_DOUBLE,all.data(),12,MPI_DOUBLE,0,mpi.comm);
        return all;
    }
#endif
public:
    void init(const GPSettings &settings, Potentials::AnyPotential &potential, const std::string &mode, bool profile=false) {
        potential_=&potential; mode_=mode;
        if (mode!="images" && mode!="atoms") throw std::invalid_argument("NEB: omp_mode must be images or atoms");
#ifdef IMRMLP_MPI
        if (mpi.size>1) { command(configure); configure_all(&settings,profile); }
#endif
        profiling_=profile;
    }
    void operator()(std::vector<Atoms> &images, int begin, int end) {
#ifdef IMRMLP_MPI
        if (mpi.size>1) { command(evaluate); evaluate_all(images,begin,end); return; }
#endif
        NEBReporting::Timer batch_timer(counter(5));
        if(profiling_) { ++statistics_[scope_*6+3];statistics_[scope_*6+4]+=end-begin; }
        NEBReporting::Timer timer(counter(0));
        evaluate_neb_images(*potential_,images,begin,end,mode_);
    }
    int workers() {
#ifdef IMRMLP_MPI
        for (;;) {
            int request; MPI_Bcast(&request,1,MPI_INT,0,mpi.comm);
            if (request==stop) {
                int status; MPI_Bcast(&status,1,MPI_INT,0,mpi.comm); return status;
            }
            if(request==release_scope) { scope_=1;continue; }
            if(request==collect_stats) { gather();continue; }
            try {
                if (request==configure) configure_all(nullptr);
                else { std::vector<Atoms> empty; evaluate_all(empty,0,0); }
            } catch (const std::exception &) {
                // phase() already delivered this error to the controller. It
                // may recover a geometry trial or send the final stop command.
            }
        }
#else
        return 0;
#endif
    }
    bool profiling() const { return profiling_; }
    void begin_release() {
        if(!profiling_) return;
#ifdef IMRMLP_MPI
        if(mpi.size>1) command(release_scope);
#endif
        scope_=1;
    }
    void write_profile(const std::string &filename, const std::array<double,4> &stages) {
        if(!profiling_) return;
        std::vector<double> all(statistics_.begin(),statistics_.end());
#ifdef IMRMLP_MPI
        if(mpi.size>1) { command(collect_stats);all=gather(); }
#endif
        std::ofstream out(filename);
        if(!out) throw std::runtime_error("Cannot open NEB profile: "+filename);
        out<<std::setprecision(17)<<"{\n\"schema_version\":1,\"mpi_ranks\":"<<mpi.size
           <<",\"omp_threads\":"<<get_omp_max_threads()<<",\"omp_mode\":\""<<mode_<<"\",\n"
             "\"timing_scope\":\"Controller stages include EFS; rank categories are overlapping across ranks. Collectives include waiting. Profile collection and file write excluded.\",\n"
             "\"controller_seconds\":{\"initialization\":"<<stages[0]<<",\"optimization\":"<<stages[1]
           <<",\"output\":"<<stages[2]<<",\"symmetry_release\":"<<stages[3]<<"},\n\"ranks\":[";
        const char *names[]={"efs_seconds","serialization_seconds","communication_wait_seconds","batches","image_evaluations","batch_wall_seconds"};
        for(int rank=0;rank<mpi.size;++rank) {
            if(rank)out<<',';
            out<<"{\"rank\":"<<rank;
            for(int scope=0;scope<2;++scope) {
                out<<",\""<<(scope?"symmetry_release":"main")<<"\":{";
                for(int k=0;k<6;++k) {if(k)out<<',';out<<'"'<<names[k]<<"\":"<<all[rank*12+scope*6+k];}
                out<<'}';
            }
            out<<'}';
        }
        out<<"]\n}\n";out.close();
        if(!out) throw std::runtime_error("Failed to write NEB profile");
    }
    void finish(int status) {
#ifdef IMRMLP_MPI
        if (mpi.size>1) { command(stop); MPI_Bcast(&status,1,MPI_INT,0,mpi.comm); }
#endif
    }
};
