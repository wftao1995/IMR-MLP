#pragma once

// GPM v1 wire format. This layer is deliberately C++17 and independent of GPML,
// Eigen and MPI so the standalone inference package uses exactly the same reader.
#include "picojson.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#ifdef USE_ZSTD
#include <zstd.h>
#endif

namespace GPM {
using Value = picojson::value;
using Object = picojson::object;
using Array = picojson::array;
using Bytes = std::vector<unsigned char>;
inline void require(bool ok, const std::string &why) {
    if (!ok) throw std::runtime_error("GPM: " + why);
}
inline const Object &object(const Value &v) {
    require(v.is<Object>(), "expected JSON object"); return v.get<Object>();
}
inline const Array &array(const Value &v) {
    require(v.is<Array>(), "expected JSON array"); return v.get<Array>();
}
inline const Value &at(const Object &v, const std::string &key) {
    auto i = v.find(key); require(i != v.end(), "missing field " + key); return i->second;
}
inline std::string string(const Value &v) {
    require(v.is<std::string>(), "expected string"); return v.get<std::string>();
}
inline int integer(const Value &v) {
    require(v.is<double>(), "expected integer"); double x = v.get<double>();
    require(std::isfinite(x) && x >= 0 && x <= INT32_MAX && std::floor(x) == x, "invalid integer");
    return static_cast<int>(x);
}
inline uint64_t decimal(const Value &v) {
    auto s = string(v); require(!s.empty() && (s.size() == 1 || s[0] != '0'), "invalid uint64");
    uint64_t n = 0;
    for (char c : s) {
        require(c >= '0' && c <= '9' && n <= (UINT64_MAX - (c-'0'))/10, "invalid uint64");
        n = n * 10 + (c-'0');
    }
    return n;
}
inline Value count(uint64_t n) { return Value(std::to_string(n)); }
inline uint64_t checked_add(uint64_t a, uint64_t b) {
    require(b <= UINT64_MAX-a, "size overflow"); return a+b;
}
inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    require(!a || b <= UINT64_MAX/a, "size overflow"); return a*b;
}
inline uint64_t align64(uint64_t x) { return checked_add(x,63) & ~uint64_t(63); }
inline void put_le(unsigned char *p, uint64_t n, unsigned bytes) {
    for (unsigned i=0;i<bytes;++i) { p[i]=static_cast<unsigned char>(n); n >>= 8; }
}
inline uint64_t get_le(const unsigned char *p, unsigned bytes) {
    uint64_t n=0; for (unsigned i=0;i<bytes;++i) n |= uint64_t(p[i]) << (8*i); return n;
}

// Incremental SHA-256, including long inputs; no crypto-library runtime required.
class SHA256 {
    std::array<uint32_t,8> h_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                              0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<unsigned char,64> buffer_{};
    uint64_t size_=0; size_t used_=0;
    static uint32_t r(uint32_t x,unsigned n) { return (x>>n)|(x<<(32-n)); }
    void transform(const unsigned char *p) {
        static constexpr uint32_t k[64]={
          0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
          0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
          0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
          0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
          0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
          0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
          0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
          0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for(int i=0;i<16;++i) w[i]=(uint32_t(p[4*i])<<24)|(uint32_t(p[4*i+1])<<16)|(uint32_t(p[4*i+2])<<8)|p[4*i+3];
        for(int i=16;i<64;++i) w[i]=w[i-16]+(r(w[i-15],7)^r(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(r(w[i-2],17)^r(w[i-2],19)^(w[i-2]>>10));
        auto a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for(int i=0;i<64;++i) {
            uint32_t t1=h+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t t2=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=h;
    }
public:
    void update(const void *data,size_t n) {
        auto p=static_cast<const unsigned char*>(data); size_+=n;
        while(n) { size_t take=std::min(n,64-used_); std::memcpy(buffer_.data()+used_,p,take);
            used_+=take;p+=take;n-=take;if(used_==64){transform(buffer_.data());used_=0;} }
    }
    std::array<unsigned char,32> finish() {
        uint64_t bits=size_*8; unsigned char pad=0x80;update(&pad,1);pad=0;
        while(used_!=56) update(&pad,1);
        unsigned char length[8];for(int i=0;i<8;++i) length[7-i]=static_cast<unsigned char>(bits>>(8*i));
        update(length,8);std::array<unsigned char,32> out{};
        for(int i=0;i<8;++i)for(int j=0;j<4;++j)out[i*4+j]=static_cast<unsigned char>(h_[i]>>(24-8*j));
        return out;
    }
};
inline std::array<unsigned char,32> digest(const void *p,size_t n) { SHA256 h;h.update(p,n);return h.finish(); }
inline std::string hex(const std::array<unsigned char,32> &d) {
    static const char digits[]="0123456789abcdef";std::string s;
    for(auto c:d){s+=digits[c>>4];s+=digits[c&15];}return s;
}
inline std::string hash(const Bytes &b) { return hex(digest(b.data(),b.size())); }
inline std::string hash(const std::string &s) { return hex(digest(s.data(),s.size())); }

// Reject duplicate keys even in unknown extension objects.
class ParseContext : public picojson::default_parse_context {
    Value *out_; unsigned depth_;
public:
    ParseContext(Value *v,unsigned depth=0):picojson::default_parse_context(v),out_(v),depth_(depth){}
    template<class I> bool parse_object_item(picojson::input<I> &in,const std::string &key) {
        auto &o=out_->get<Object>();if(depth_>=64 || o.count(key))return false;
        ParseContext c(&o[key],depth_+1);return picojson::_parse(c,in);
    }
    template<class I> bool parse_array_item(picojson::input<I> &in,size_t) {
        if(depth_>=64)return false;auto &a=out_->get<Array>();a.emplace_back();
        ParseContext c(&a.back(),depth_+1);return picojson::_parse(c,in);
    }
};
inline Value parse(const std::string &s) {
    for(size_t i=0;i<s.size();) {
        unsigned char lead=s[i++];if(lead<0x80)continue;
        unsigned tails;uint32_t code;
        if(lead>=0xc2&&lead<=0xdf){tails=1;code=lead&31;}
        else if(lead>=0xe0&&lead<=0xef){tails=2;code=lead&15;}
        else if(lead>=0xf0&&lead<=0xf4){tails=3;code=lead&7;}
        else {require(false,"invalid UTF-8");return {};}
        require(tails<=s.size()-i,"truncated UTF-8");
        for(unsigned k=0;k<tails;++k){unsigned char next=s[i++];require((next&0xc0)==0x80,"invalid UTF-8 continuation");code=(code<<6)|(next&63);}
        require(code>=(tails==1?0x80u:tails==2?0x800u:0x10000u)&&code<=0x10ffffu&&!(code>=0xd800u&&code<=0xdfffu),"invalid UTF-8 code point");
    }
    Value v;ParseContext c(&v);std::string error;
    auto end=picojson::_parse(c,s.begin(),s.end(),&error);
    require(error.empty() && std::all_of(end,s.end(),[](char x){return x==' '||x=='\n'||x=='\r'||x=='\t';}),"invalid JSON: "+error);
    return v;
}
inline unsigned width(const std::string &dtype) {
    if(dtype=="f64"||dtype=="i64"||dtype=="u64")return 8;
    if(dtype=="i32"||dtype=="u32")return 4;
    if(dtype=="u8")return 1;
    throw std::runtime_error("GPM: unsupported dtype "+dtype);
}
struct Block {
    std::string id,dtype,codec,sha256;
    std::vector<uint64_t> shape;
    uint64_t offset=0,stored_size=0,raw_size=0;
    Object json() const {
        Array dims;for(auto n:shape)dims.emplace_back(count(n));
        return {{"id",Value(id)},{"dtype",Value(dtype)},{"shape",Value(dims)},
          {"codec",Value(codec)},{"sha256",Value(sha256)},{"offset",count(offset)},
          {"stored_size",count(stored_size)},{"raw_size",count(raw_size)}};
    }
};
inline bool is_file(const std::string &path) {
    std::ifstream in(path,std::ios::binary);char h[8]{};in.read(h,8);
    return in.gcount()==8 && std::memcmp(h,"GPMLMODL",8)==0;
}
inline bool extension(const std::string &path) { return std::filesystem::path(path).extension()==".gpm"; }
class Reader {
    struct Source { std::ifstream in;std::mutex mutex;explicit Source(const std::string&p):in(p,std::ios::binary){} };
    std::shared_ptr<Source> source_;
    uint64_t data_offset_=0;
public:
    Object manifest;
    std::map<std::string,Block> blocks;
    explicit Reader(const std::string &path):source_(std::make_shared<Source>(path)) {
        auto &in=source_->in;require(bool(in),"cannot open "+path);
        in.seekg(0,std::ios::end);auto end=in.tellg();require(end>=64,"truncated preamble");
        uint64_t size=static_cast<uint64_t>(end);in.seekg(0);
        std::array<unsigned char,64> h{};in.read(reinterpret_cast<char*>(h.data()),64);
        require(std::memcmp(h.data(),"GPMLMODL",8)==0,"invalid magic");
        require(get_le(h.data()+8,2)==1,"unsupported major version");
        require(get_le(h.data()+12,4)==0,"unsupported required flags");
        uint64_t length=get_le(h.data()+16,8);data_offset_=get_le(h.data()+24,8);
        require(length<=64*1024*1024 && data_offset_==align64(checked_add(64,length)) && data_offset_<=size,"invalid header size");
        std::string json(static_cast<size_t>(length),'\0');in.read(json.data(),json.size());
        auto d=digest(json.data(),json.size());require(bool(in)&&std::equal(d.begin(),d.end(),h.begin()+32),"header checksum mismatch");
        manifest=object(parse(json));require(string(at(manifest,"schema"))=="gpml.model","unknown schema");
        if(manifest.count("required_features"))require(array(manifest.at("required_features")).empty(),"unsupported required features");
        std::vector<std::pair<uint64_t,uint64_t>> intervals;
        for(const auto &entry:array(at(manifest,"blocks"))) {
            const auto &o=object(entry);Block b;
            b.id=string(at(o,"id"));b.dtype=string(at(o,"dtype"));b.codec=string(at(o,"codec"));b.sha256=string(at(o,"sha256"));
            b.offset=decimal(at(o,"offset"));b.stored_size=decimal(at(o,"stored_size"));b.raw_size=decimal(at(o,"raw_size"));
            uint64_t elements=1;for(const auto &x:array(at(o,"shape"))){auto n=decimal(x);elements=checked_mul(elements,n);b.shape.push_back(n);}
            require(checked_mul(elements,width(b.dtype))==b.raw_size,"block shape/size mismatch: "+b.id);
            require(!b.id.empty() && b.offset%64==0 && b.offset<=size-data_offset_ && b.stored_size<=size-data_offset_-b.offset,"invalid block bounds: "+b.id);
            require(b.sha256.size()==64 && std::all_of(b.sha256.begin(),b.sha256.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}),"invalid checksum");
            require(b.codec!="none" || b.stored_size==b.raw_size,"invalid uncompressed block size");
            require(blocks.emplace(b.id,b).second,"duplicate block id");
            if(b.stored_size)intervals.emplace_back(b.offset,checked_add(b.offset,b.stored_size));
        }
        std::sort(intervals.begin(),intervals.end());
        for(size_t i=1;i<intervals.size();++i)require(intervals[i-1].second<=intervals[i].first,"overlapping blocks");
    }
    const Block &info(const std::string &id) const {
        auto i=blocks.find(id);require(i!=blocks.end(),"missing block "+id);return i->second;
    }
    Bytes stored(const std::string &id) const {
        const auto &b=info(id);require(b.stored_size<=std::numeric_limits<size_t>::max(),"stored block too large");Bytes bytes(static_cast<size_t>(b.stored_size));
        std::lock_guard<std::mutex> lock(source_->mutex);auto &in=source_->in;in.clear();in.seekg(data_offset_+b.offset);in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());require(bool(in),"truncated block "+id);return bytes;
    }
    Bytes read(const std::string &id,uint64_t limit=uint64_t(8)<<30) const {
        const auto &b=info(id);require(b.raw_size<=limit && b.stored_size<=limit,"block exceeds read limit: "+id);
        require(b.codec=="none" || b.codec=="zstd","unsupported codec "+b.codec);
#ifndef USE_ZSTD
        require(b.codec!="zstd","zstd support is not enabled");
#endif
        Bytes stored(static_cast<size_t>(b.stored_size));
        { std::lock_guard<std::mutex> lock(source_->mutex);auto &in=source_->in;in.clear();
          in.seekg(data_offset_+b.offset);in.read(reinterpret_cast<char*>(stored.data()),stored.size());require(bool(in),"truncated block "+id); }
        Bytes raw;
        if(b.codec=="none")raw=std::move(stored);
#ifdef USE_ZSTD
        else {raw.resize(static_cast<size_t>(b.raw_size));auto n=ZSTD_decompress(raw.data(),raw.size(),stored.data(),stored.size());
          require(!ZSTD_isError(n)&&n==raw.size(),"invalid zstd block "+id);}
#endif
        require(hash(raw)==b.sha256,"block checksum mismatch: "+id);return raw;
    }
    void validate() const { for(const auto &b:blocks)read(b.first); }
};

class Writer {
    struct Data {Block block;std::function<Bytes()> read;bool encoded=false;};
    std::map<std::string,Data> data_;
public:
    Object manifest{{"schema",Value("gpml.model")},{"units",Value(Object{{"length",Value("angstrom")},{"energy",Value("eV")},{"charge",Value("e")},{"stress",Value("virial_eV")}})},
        {"species",Value(Array{})},{"root_model",Value("root")},{"models",Value(Object{})}};
    void add(std::string id,std::string dtype,std::vector<uint64_t> shape,Bytes bytes,std::string codec="none") {
        uint64_t n=1;for(auto x:shape)n=checked_mul(n,x);require(checked_mul(n,width(dtype))==bytes.size(),"incorrect block shape");
        require(!id.empty()&&!data_.count(id),"duplicate block "+id);
        auto p=std::make_shared<Bytes>(std::move(bytes));Block b;
        b.id=id;b.dtype=std::move(dtype);b.shape=std::move(shape);b.codec=std::move(codec);b.raw_size=p->size();b.sha256=hash(*p);
        data_.emplace(std::move(id),Data{std::move(b),[p]{return *p;}});
    }
    void preserve(std::shared_ptr<Reader> source) {
        manifest=source->manifest;
        for(const auto &item:source->blocks){auto id=item.first;data_[id]={item.second,[source,id]{
            const auto &codec=source->info(id).codec;
            if(codec=="none")source->read(id);
#ifdef USE_ZSTD
            else if(codec=="zstd")source->read(id);
#endif
            return source->stored(id);},true};}
    }
    std::string block_hash(const std::string &id) const {return data_.at(id).block.sha256;}
    void save(const std::string &path) const {
        auto parent=std::filesystem::path(path).parent_path();if(parent.empty())parent=".";
        std::string temp=(parent/("."+std::filesystem::path(path).filename().string()+".XXXXXX")).string();
        int fd=::mkstemp(temp.data());require(fd>=0,"cannot create temporary output for "+path);::close(fd);
        // Stage data first. JSON length and offsets can then be determined without
        // holding compressed copies of every block at once.
        std::string payload=temp+".data";
        try {
            std::ofstream out(payload,std::ios::binary|std::ios::trunc);require(bool(out),"cannot stage payload");
            Array directory;uint64_t cursor=0;
            const char zeros[64]{};
            for(const auto &entry:data_) {
                Block b=entry.second.block;Bytes raw=entry.second.read();if(!entry.second.encoded)require(raw.size()==b.raw_size&&hash(raw)==b.sha256,"source block changed");
                Bytes stored;
                if(entry.second.encoded||b.codec=="none")stored=std::move(raw);
                else if(b.codec=="zstd") {
#ifdef USE_ZSTD
                    stored.resize(ZSTD_compressBound(raw.size()));auto n=ZSTD_compress(stored.data(),stored.size(),raw.data(),raw.size(),3);
                    require(!ZSTD_isError(n),"compression failed");stored.resize(n);
#else
                    throw std::runtime_error("GPM: zstd support is not enabled");
#endif
                } else throw std::runtime_error("GPM: unsupported codec "+b.codec);
                b.offset=align64(cursor);out.write(zeros,b.offset-cursor);b.stored_size=stored.size();
                out.write(reinterpret_cast<const char*>(stored.data()),stored.size());cursor=checked_add(b.offset,b.stored_size);
                directory.emplace_back(b.json());
            }
            out.close();require(bool(out),"payload write failed");
            Object header=manifest;header["blocks"]=Value(directory);std::string json=Value(header).serialize(true);
            require(json.size()<=64*1024*1024,"header exceeds limit");
            std::array<unsigned char,64> pre{};std::memcpy(pre.data(),"GPMLMODL",8);put_le(pre.data()+8,1,2);
            put_le(pre.data()+16,json.size(),8);uint64_t start=align64(64+json.size());put_le(pre.data()+24,start,8);
            auto d=digest(json.data(),json.size());std::copy(d.begin(),d.end(),pre.begin()+32);
            std::ofstream file(temp,std::ios::binary|std::ios::trunc);file.write(reinterpret_cast<char*>(pre.data()),64);file.write(json.data(),json.size());file.write(zeros,start-64-json.size());
            std::ifstream input(payload,std::ios::binary);std::array<char,65536> chunk{};
            while(input){input.read(chunk.data(),chunk.size());file.write(chunk.data(),input.gcount());}
            require(input.eof()&&!input.bad(),"payload read failed");file.close();require(bool(file),"output write failed");
            Reader verified(temp);for(const auto &entry:data_)if(!entry.second.encoded)verified.read(entry.first);fd=::open(temp.c_str(),O_RDONLY);require(fd>=0,"cannot synchronize output");
            int synced=::fsync(fd);::close(fd);require(synced==0,"output synchronization failed");
            std::filesystem::rename(temp,path);
            fd=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);if(fd>=0){::fsync(fd);::close(fd);}
            std::filesystem::remove(payload);
        } catch(...) {std::error_code ec;std::filesystem::remove(temp,ec);std::filesystem::remove(payload,ec);throw;}
    }
};
inline Bytes encode_f64(const double *p,size_t n) {
    static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"GPM requires IEEE FP64");
    Bytes out(checked_mul(n,8));for(size_t i=0;i<n;++i){uint64_t x;std::memcpy(&x,p+i,8);put_le(out.data()+8*i,x,8);}return out;
}
inline std::vector<double> decode_f64(const Reader&r,const std::string&id) {
    require(r.info(id).dtype=="f64","expected FP64 block");auto b=r.read(id);std::vector<double> out(b.size()/8);
    for(size_t i=0;i<out.size();++i){uint64_t x=get_le(b.data()+8*i,8);std::memcpy(&out[i],&x,8);}return out;
}
} // namespace GPM
