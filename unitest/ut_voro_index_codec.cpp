#include <catch2/catch_all.hpp>
#include "base/voro_index_codec.h"
#include "base/neighborhood_voro.h"
#include <map>
#include <set>
using C32=VoroIndex::Codec<std::uint32_t,4>;
using C64=VoroIndex::Codec<std::uint64_t,5>;
TEST_CASE("Voronoi width codecs preserve existing representable values and text", "[voro][codec]") {
    std::vector<int> histogram{0,0,0,0,6,0,8,0,0,0,0,0};
    C32 a(histogram);C64 b(histogram);
    REQUIRE(a.packed());REQUIRE(b.packed());
    std::uint64_t old=0;
    for(auto i=histogram.size();i>2;--i)old=(old<<5)+histogram[i-1];
    REQUIRE(b.index==old);
    REQUIRE(a.tostring(12)==b.tostring(12));
    REQUIRE(b.tostring(12)=="0-0-0-6-0-8-0-0-0-0-0");
    REQUIRE(sizeof(VoroIndex::NativeUInt)==sizeof(void*));
}
TEST_CASE("Voronoi overflow counters and long histograms retain complete identity", "[voro][codec]") {
    std::vector<std::vector<int>> cases{{0,0},{0,0,16},{0,0,32},{0,0,0,1},std::vector<int>(20),std::vector<int>(20)};
    cases[4][19]=1;cases[5][19]=2;
    std::set<C32> a;std::set<C64> b;
    for(const auto& h:cases) {
        a.emplace(h);b.emplace(h);
        REQUIRE(C32(h).tostring(3)==C64(h).tostring(3));
    }
    REQUIRE(a.size()==cases.size());REQUIRE(b.size()==cases.size());
    REQUIRE_FALSE(C32(cases[1]).packed());REQUIRE(C64(cases[1]).packed());
    REQUIRE_FALSE(C64(cases[2]).packed());
    auto ia=a.begin();auto ib=b.begin();
    for(;ia!=a.end();++ia,++ib)REQUIRE(ia->tostring(20)==ib->tostring(20));
    std::vector<int> zeros(100);REQUIRE_FALSE(C64(zeros)<C64({0,0}));REQUIRE_FALSE(C64({0,0})<C64(zeros));
    REQUIRE_THROWS_AS(C32({}),std::invalid_argument);
    REQUIRE(C32({0,1,0}).tostring(3)==C32({0,0,0}).tostring(3));
    REQUIRE_THROWS_AS(C64({0,0,-1}),std::invalid_argument);
}

TEST_CASE("Voronoi species keys order full indices before chemical types", "[voro][codec]") {
    std::vector<int> first(20),second(20);first[19]=1;second[19]=2;
    const voro_index_bond a(first,2),b(first,3),c(second,0);
    REQUIRE(a<b);REQUIRE_FALSE(b<a);REQUIRE(b<c);
    REQUIRE(a.tostring(20)==voro_index(first).tostring(20));
    std::map<voro_index_bond,int> counts;counts[a]++;counts[b]++;counts[c]++;
    REQUIRE(counts.size()==3);
}
