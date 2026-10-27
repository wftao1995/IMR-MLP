#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace VoroIndex {
// Fast packed representation when possible; otherwise retain the complete
// histogram. Ordering compares highest face-order count first, matching the
// old packed integer order without truncating large counts or high orders.
template<class UInt,int Pack> struct Codec {
    static_assert(std::is_unsigned_v<UInt> && Pack>0 && Pack<std::numeric_limits<UInt>::digits);
    static constexpr int bits=std::numeric_limits<UInt>::digits;
    static constexpr int capacity=bits/Pack;
    static constexpr UInt mask=(UInt(1)<<Pack)-1;
    UInt index=0;
    std::vector<int> overflow;

    explicit Codec(const std::vector<int>& histogram) {
        if(histogram.size()<2)
            throw std::invalid_argument("Voronoi index requires at least two counters");
        // Legacy index semantics omit zero/one-edge faces. Edge filtering can
        // populate these slots, so do not reject them or change their meaning.
        size_t end=histogram.size();
        for(int count:histogram)if(count<0)throw std::invalid_argument("Negative Voronoi counter");
        while(end>2 && histogram[end-1]==0)--end;
        bool fits=end-2<=size_t(capacity);
        for(size_t i=2;i<end;++i)if(static_cast<std::uintmax_t>(histogram[i])>mask)fits=false;
        if(!fits) {overflow.assign(histogram.begin()+2,histogram.begin()+end);return;}
        for(size_t i=end;i>2;--i)index=(index<<Pack)|UInt(histogram[i-1]);
    }
    bool packed()const{return overflow.empty();}
    size_t counters()const {
        if(!packed())return overflow.size();
        UInt rest=index;size_t count=0;
        while(rest){rest>>=Pack;++count;}return count;
    }
    int counter(size_t i)const {
        if(!packed())return i<overflow.size() ? overflow[i] : 0;
        return i<size_t(capacity) ? int((index>>(i*Pack))&mask) : 0;
    }
    bool operator<(const Codec& other)const {
        if(packed() && other.packed())return index<other.index;
        for(size_t n=std::max(counters(),other.counters());n>0;--n) {
            const int a=counter(n-1),b=other.counter(n-1);
            if(a!=b)return a<b;
        }
        return false;
    }
    std::string tostring(int edges)const {
        if(edges<2)throw std::invalid_argument("Invalid Voronoi output order");
        // A caller's short display bound must not hide an overflow tail.
        const size_t count=std::max(size_t(edges-2),counters());
        std::string result="0";
        for(size_t i=0;i<count;++i)result+="-"+std::to_string(counter(i));
        return result;
    }
};
using NativeUInt=std::conditional_t<sizeof(void*)==4,std::uint32_t,std::uint64_t>;
inline constexpr int native_pack=sizeof(void*)==4 ? 4 : 5;
using Native=Codec<NativeUInt,native_pack>;
}
