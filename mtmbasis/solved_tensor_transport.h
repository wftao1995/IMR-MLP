#pragma once
#include "mtm.h"
#include <climits>
#include <stdexcept>

namespace mtmbasis_detail {
// Only fields established by solve() are transported. Pool IDs are assigned
// later and must never be read here.
inline std::vector<int> pack_solved_tensors(const std::vector<alpha_index_type>& tensors) {
    std::vector<int> data;
    auto size=[](size_t n) {if(n>INT_MAX)throw std::length_error("Tensor packet exceeds integer size");return int(n);};
    data.push_back(size(tensors.size()));
    auto append=[&](const auto& matrix){for(Eigen::Index i=0;i<matrix.size();++i)data.push_back(matrix.data()[i]);};
    for(const auto& t:tensors) {
        const int n=size(t.size());
        if(!t.solved || t.second.rows()!=n || t.second.cols()!=n ||
           t.stdform.rows()!=n+1 || t.stdform.cols()!=n+1 ||
           t.musorted.size()!=n || t.nvsorted.size()!=n || t.diag.size()!=n)
            throw std::invalid_argument("Only complete solved tensors can be transported");
        data.insert(data.end(),{n,t.mu,t.nv,size(t.canonical_orders.size())});
        append(t.first);append(t.second);append(t.stdform);
        append(t.musorted);append(t.nvsorted);append(t.diag);
        for(const auto& order:t.canonical_orders) {
            if(order.size()!=size_t(n))throw std::invalid_argument("Invalid canonical tensor order");
            data.insert(data.end(),order.begin(),order.end());
        }
    }
    return data;
}
inline std::vector<alpha_index_type> unpack_solved_tensors(const std::vector<int>& data) {
    size_t offset=0;
    auto next=[&]() {if(offset==data.size())throw std::invalid_argument("Truncated solved tensor packet");return data[offset++];};
    const int count=next();
    if(count<0 || size_t(count)>data.size())throw std::invalid_argument("Invalid solved tensor count");
    std::vector<alpha_index_type> result;result.reserve(count);
    auto read=[&](auto& matrix){for(Eigen::Index i=0;i<matrix.size();++i)matrix.data()[i]=next();};
    for(int k=0;k<count;++k) {
        const int n=next(),mu=next(),nv=next(),orders=next();
        if(n<0 || orders<1 || size_t(n)>data.size() ||
           (size_t(n)+1)*(size_t(n)+1)>data.size()-offset ||
           size_t(orders)>data.size()-offset)
            throw std::invalid_argument("Invalid solved tensor dimensions");
        alpha_index_type t;t.first.resize(n);t.second.resize(n,n);t.stdform.resize(n+1,n+1);
        t.musorted.resize(n);t.nvsorted.resize(n);t.diag.resize(n);
        read(t.first);read(t.second);read(t.stdform);read(t.musorted);read(t.nvsorted);read(t.diag);
        t.canonical_orders.resize(orders);
        for(auto& order:t.canonical_orders) {
            order.resize(n);std::vector<bool> seen(n,false);
            for(int& i:order) {i=next();if(i<0 || i>=n || seen[i])throw std::invalid_argument("Invalid canonical tensor permutation");seen[i]=true;}
        }
        t.mu=mu;t.nv=nv;t.solved=true;result.push_back(std::move(t));
    }
    if(offset!=data.size())throw std::invalid_argument("Trailing solved tensor packet data");
    return result;
}
inline void broadcast_solved_tensors(std::vector<alpha_index_type>& tensors) {
    std::vector<int> packet;
    if(mpi.rank==0)packet=pack_solved_tensors(tensors);
    const size_t base=size_t(INT_MAX)+1;
    int lengths[2]={int(packet.size()/base),int(packet.size()%base)};
    MPI_Bcast(lengths,2,MPI_INT,0,mpi.comm);
    if(mpi.rank!=0)packet.resize(size_t(lengths[0])*base+size_t(lengths[1]));
    for(size_t offset=0;offset<packet.size();) {
        const int count=int(std::min<size_t>(INT_MAX,packet.size()-offset));
        MPI_Bcast(packet.data()+offset,count,MPI_INT,0,mpi.comm);offset+=count;
    }
    if(mpi.rank!=0)tensors=unpack_solved_tensors(packet);
}
}
