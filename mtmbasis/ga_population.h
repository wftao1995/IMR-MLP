#pragma once
#include "mtm.h"
#include <set>
#include <random>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mtmbasis_detail {
struct GAIndividual {
    std::vector<int> genes;
    std::pair<int,int> fitness;
};
inline bool ga_individual_better(const GAIndividual& a,const GAIndividual& b,double w1,double w2) {
    if(weighted_fitness_better(a.fitness,b.fitness,w1,w2)) return true;
    if(weighted_fitness_better(b.fitness,a.fitness,w1,w2)) return false;
    return a.genes<b.genes;
}
inline void update_ga_elites(std::vector<GAIndividual>& archive,const std::vector<GAIndividual>& candidates,
    size_t count,double w1,double w2) {
    archive.insert(archive.end(),candidates.begin(),candidates.end());
    std::stable_sort(archive.begin(),archive.end(),[&](const auto& a,const auto& b){return ga_individual_better(a,b,w1,w2);});
    std::set<std::vector<int>> seen;std::vector<GAIndividual> selected;
    for(auto& individual:archive) if(seen.insert(individual.genes).second) {
        selected.push_back(std::move(individual));if(selected.size()==count)break;
    }
    if(count==0) selected.clear();
    archive=std::move(selected);
}
inline std::mt19937 ga_random_stream(uint32_t seed,int rank) {
    std::seed_seq sequence{seed,static_cast<uint32_t>(rank)};
    return std::mt19937(sequence);
}
struct DiverseGAPopulation {
    std::vector<GAIndividual> members;
    size_t unique_count=0,evaluations=0;
};
// The evaluator receives each proposal by mutable reference and returns its
// actual repaired genotype. No fitness memoization bypasses its side effects.
template<class Evaluate>
DiverseGAPopulation diverse_ga_population(const GAIndividual& anchor,const std::vector<int>& upper,
    const std::vector<int>& variable,size_t requested,int max_sites,std::mt19937& random,Evaluate&& evaluate) {
    if(requested==0 || max_sites<=0 || upper.size()!=anchor.genes.size())
        throw std::invalid_argument("Invalid diverse GA population request");
    for(int i:variable) if(i<0 || size_t(i)>=upper.size() || upper[i]<=0)
        throw std::invalid_argument("Invalid variable GA gene");
    DiverseGAPopulation result;result.members.push_back(anchor);
    std::set<std::vector<int>> seen{anchor.genes};
    auto remember=[&](std::vector<int> proposal) {
        if(seen.count(proposal)) return;
        const auto fitness=evaluate(proposal);++result.evaluations;
        if(seen.insert(proposal).second) result.members.push_back({std::move(proposal),fitness});
    };
    if(requested>std::numeric_limits<size_t>::max()/64) throw std::length_error("GA population is too large");
    const size_t budget=std::max<size_t>(32,requested*16);
    for(size_t attempt=0;attempt<budget && result.members.size()<requested && !variable.empty();++attempt) {
        auto proposal=anchor.genes;
        if(attempt<budget/2) {
            auto sites=variable;std::shuffle(sites.begin(),sites.end(),random);
            const int mutations=std::uniform_int_distribution<int>(1,std::min<int>(max_sites,sites.size()))(random);
            for(int j=0;j<mutations;++j) {
                const int i=sites[j];
                if(proposal[i]>=0 && proposal[i]<=upper[i]) {
                    int replacement=std::uniform_int_distribution<int>(0,upper[i]-1)(random);
                    proposal[i]=replacement>=proposal[i] ? replacement+1 : replacement;
                } else proposal[i]=std::uniform_int_distribution<int>(0,upper[i])(random);
            }
        } else for(int i:variable)proposal[i]=std::uniform_int_distribution<int>(0,upper[i])(random);
        remember(std::move(proposal));
    }
    // Enumerate a bounded prefix as a deterministic fallback. Tiny spaces are
    // exhausted, while large repair-collapsed spaces cannot cause an endless loop.
    auto proposal=anchor.genes;for(int i:variable)proposal[i]=0;
    for(size_t attempt=0;attempt<requested*64 && result.members.size()<requested;++attempt) {
        remember(proposal);bool advanced=false;
        for(int i:variable) {
            if(proposal[i]<upper[i]) {++proposal[i];advanced=true;break;}
            proposal[i]=0;
        }
        if(!advanced) break;
    }
    result.unique_count=result.members.size();
    for(size_t i=0;result.members.size()<requested;++i) result.members.push_back(result.members[i%result.unique_count]);
    return result;
}
}
