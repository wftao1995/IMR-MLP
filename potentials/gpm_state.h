#pragma once
#include "tools/gpm_container.h"
#include <Eigen/Core>
#include <type_traits>

namespace GPM {
// One contiguous array per structure property/dtype for the whole collection.
class NumericPool {
    struct Data {std::string dtype;Bytes bytes;};
    Writer &writer_;std::string prefix_;std::map<std::string,Data> data_;
public:
    NumericPool(Writer &writer,std::string prefix):writer_(writer),prefix_(std::move(prefix)){}
    Value add(const std::string &key,const std::string &dtype,const std::vector<uint64_t> &shape,Bytes bytes) {
        Array dimensions;for(auto size:shape)dimensions.emplace_back(count(size));
        if(bytes.empty())return Value(Object{{"dtype",Value(dtype)},{"shape",Value(dimensions)}});
        const auto name=prefix_+"/arrays/"+key.substr(0,key.find('/'))+"/"+dtype;
        auto &data=data_[name];data.dtype=dtype;const auto start=data.bytes.size()/width(dtype);
        data.bytes.insert(data.bytes.end(),bytes.begin(),bytes.end());
        return Value(Object{{"block",Value(name)},{"start",count(start)},{"shape",Value(dimensions)}});
    }
    void finish() {
        for(auto &entry:data_){auto &data=entry.second;const auto size=data.bytes.size()/width(data.dtype);
            writer_.add(entry.first,data.dtype,{size},std::move(data.bytes));}
    }
};
using BlockCache=std::map<std::string,Bytes>;
// Named state fields. Numeric arrays have a portable row-major representation;
// this is also used by AL and configuration records, never by object memcpy.
class State {
    Writer *writer_=nullptr;
    NumericPool *pool_=nullptr;
    BlockCache *cache_=nullptr;
    const Reader *reader_=nullptr;
    std::string prefix_;
    Object fields_;
    template<class T> static std::string dtype() {
        if constexpr(std::is_same_v<T,double>)return "f64";
        else if constexpr(sizeof(T)==8)return std::is_signed_v<T>?"i64":"u64";
        else if constexpr(sizeof(T)==4)return std::is_signed_v<T>?"i32":"u32";
        else return "u8";
    }
    template<class T> Value numeric(const std::string &key,const T *p,size_t n,std::vector<uint64_t> dims) {
        static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>);
        Bytes bytes(checked_mul(n,sizeof(T)));
        for(size_t i=0;i<n;++i){uint64_t bits=0;
            if constexpr(std::is_same_v<T,double>)std::memcpy(&bits,p+i,8);
            else bits=static_cast<uint64_t>(p[i]);
            put_le(bytes.data()+i*sizeof(T),bits,sizeof(T));}
        if(pool_)return pool_->add(key,dtype<T>(),dims,std::move(bytes));
        if(bytes.empty()){Array shape;for(auto size:dims)shape.emplace_back(count(size));return Value(Object{{"dtype",Value(dtype<T>())},{"shape",Value(shape)}});}
        std::string id=prefix_+"/"+key;writer_->add(id,dtype<T>(),std::move(dims),std::move(bytes));
        return Value(Object{{"block",Value(id)}});
    }
    template<class T> std::vector<T> numeric(const Value &value,std::vector<uint64_t> *dims=nullptr) {
        const auto &reference=object(value);std::vector<uint64_t> shape;
        if(reference.count("shape"))for(const auto &size:array(at(reference,"shape")))shape.push_back(decimal(size));
        if(!reference.count("block")) {
            uint64_t count=1;for(auto size:shape)count=checked_mul(count,size);
            require(!shape.empty()&&count==0&&string(at(reference,"dtype"))==dtype<T>(),"invalid empty array");
            if(dims)*dims=shape;return {};
        }
        const auto id=string(at(reference,"block"));const auto &info=reader_->info(id);
        require(info.dtype==dtype<T>(),"state field dtype mismatch: "+id);
        if(shape.empty())shape=info.shape;
        uint64_t count=1;for(auto size:shape)count=checked_mul(count,size);
        const uint64_t start=reference.count("start")?decimal(at(reference,"start")):0;
        require(start<=info.raw_size/sizeof(T)&&count<=info.raw_size/sizeof(T)-start,"array slice out of bounds: "+id);
        if(dims)*dims=shape;
        Bytes owned;const Bytes *bytes;
        if(cache_){auto it=cache_->find(id);if(it==cache_->end())it=cache_->emplace(id,reader_->read(id)).first;bytes=&it->second;}
        else {owned=reader_->read(id);bytes=&owned;}
        std::vector<T> result(static_cast<size_t>(count));
        for(size_t i=0;i<result.size();++i){uint64_t bits=get_le(bytes->data()+(start+i)*sizeof(T),sizeof(T));
            if constexpr(std::is_same_v<T,double>)std::memcpy(&result[i],&bits,8);
            else if constexpr(std::is_signed_v<T>){using U=std::make_unsigned_t<T>;U x=static_cast<U>(bits);std::memcpy(&result[i],&x,sizeof(T));}
            else result[i]=static_cast<T>(bits);}
        return result;
    }
    template<class T> Value encode(const std::string &key,const T &x) {
        if constexpr(std::is_same_v<T,std::string> || std::is_same_v<T,bool>)return Value(x);
        else if constexpr(std::is_same_v<T,double>) {
            if(std::isfinite(x)&&!(x==0&&std::signbit(x)))return Value(x);
            return Value(Object{{"float",Value(std::isnan(x)?"nan":x==0?"-0":x>0?"inf":"-inf")}});
        }
        else if constexpr(std::is_enum_v<T>)return Value(double(static_cast<int>(x)));
        else if constexpr(std::is_integral_v<T> && sizeof(T)<8)return Value(double(x));
        else if constexpr(std::is_integral_v<T>)return Value(std::to_string(x));
        else static_assert(sizeof(T)==0,"unsupported GPM state field");
    }
    template<class T> void decode(const Value &v,T &x) {
        if constexpr(std::is_same_v<T,std::string>)x=string(v);
        else if constexpr(std::is_same_v<T,bool>){require(v.is<bool>(),"expected bool");x=v.get<bool>();}
        else if constexpr(std::is_same_v<T,double>){
            if(v.is<double>())x=v.get<double>();
            else {const auto text=string(at(object(v),"float"));
                require(text=="nan"||text=="inf"||text=="-inf"||text=="-0","invalid float scalar");
                x=text=="nan"?std::numeric_limits<double>::quiet_NaN():text=="inf"?std::numeric_limits<double>::infinity():text=="-inf"?-std::numeric_limits<double>::infinity():-0.;}
        }
        else if constexpr(std::is_enum_v<T>){int i;decode(v,i);x=static_cast<T>(i);}
        else if constexpr(std::is_integral_v<T> && sizeof(T)<8){require(v.is<double>(),"expected integer");double d=v.get<double>();
            require(std::isfinite(d)&&std::floor(d)==d&&d>=std::numeric_limits<T>::lowest()&&d<=std::numeric_limits<T>::max(),"integer out of range");x=static_cast<T>(d);}
        else if constexpr(std::is_unsigned_v<T>)x=decimal(v);
        else {auto s=string(v);size_t used=0;auto n=std::stoll(s,&used);require(used==s.size(),"invalid integer");x=n;}
    }
    template<class T> Value encode(const std::string &key,const std::vector<T> &x) {
        if constexpr(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>)return numeric(key,x.data(),x.size(),{x.size()});
        else {Array a;for(size_t i=0;i<x.size();++i)a.push_back(encode(key+"/"+std::to_string(i),T(x[i])));return Value(a);}
    }
    template<class T> void decode(const Value &v,std::vector<T> &x) {
        if constexpr(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>) {std::vector<uint64_t> shape;x=numeric<T>(v,&shape);require(shape.size()==1,"expected vector");}
        else {const auto &a=array(v);x.resize(a.size());for(size_t i=0;i<a.size();++i){T tmp{};decode(a[i],tmp);x[i]=std::move(tmp);}}
    }
    template<class T,int R,int C,int O,int MR,int MC>
    Value encode(const std::string &key,const std::vector<Eigen::Matrix<T,R,C,O,MR,MC>> &values) {
        const Eigen::Index rows=values.empty()?(R==Eigen::Dynamic?0:R):values.front().rows();
        const Eigen::Index cols=values.empty()?(C==Eigen::Dynamic?0:C):values.front().cols();
        for(const auto &m:values)if(m.rows()!=rows||m.cols()!=cols) {
            Array records;for(size_t i=0;i<values.size();++i)records.push_back(encode(key+"/"+std::to_string(i),values[i]));return Value(records);
        }
        std::vector<T> flat;flat.reserve(checked_mul(values.size(),checked_mul(rows,cols)));
        for(const auto &m:values)for(Eigen::Index i=0;i<rows;++i)for(Eigen::Index j=0;j<cols;++j)flat.push_back(m(i,j));
        return numeric(key,flat.data(),flat.size(),{values.size(),uint64_t(rows),uint64_t(cols)});
    }
    template<class T,int R,int C,int O,int MR,int MC>
    void decode(const Value &value,std::vector<Eigen::Matrix<T,R,C,O,MR,MC>> &values) {
        if(value.is<Array>()){const auto &records=array(value);values.resize(records.size());for(size_t i=0;i<records.size();++i)decode(records[i],values[i]);return;}
        std::vector<uint64_t> shape;const auto flat=numeric<T>(value,&shape);
        require(shape.size()==3&&shape[1]<=INT32_MAX&&shape[2]<=INT32_MAX,"invalid matrix vector dimensions");
        require((R==Eigen::Dynamic||shape[1]==R)&&(C==Eigen::Dynamic||shape[2]==C),"fixed matrix vector dimensions mismatch");
        values.resize(shape[0]);size_t cursor=0;
        for(auto &m:values){m.resize(shape[1],shape[2]);for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)m(i,j)=flat[cursor++];}
    }
    template<class T,int R,int C,int O,int MR,int MC>
    Value encode(const std::string &key,const Eigen::Matrix<T,R,C,O,MR,MC> &m) {
        std::vector<T> v;v.reserve(m.size());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)v.push_back(m(i,j));
        return numeric(key,v.data(),v.size(),{uint64_t(m.rows()),uint64_t(m.cols())});
    }
    template<class T,int R,int C,int O,int MR,int MC>
    void decode(const Value &v,Eigen::Matrix<T,R,C,O,MR,MC> &m) {
        std::vector<uint64_t> shape;auto a=numeric<T>(v,&shape);require(shape.size()==2 && shape[0]<=INT32_MAX && shape[1]<=INT32_MAX,"invalid matrix dimensions");
        require((R==Eigen::Dynamic||shape[0]==R)&&(C==Eigen::Dynamic||shape[1]==C),"fixed matrix dimensions mismatch");
        m.resize(shape[0],shape[1]);size_t p=0;for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)m(i,j)=a[p++];
    }
    template<class T,size_t N> Value encode(const std::string &key,const T (&x)[N]) {
        if constexpr(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>)return numeric(key,x,N,{N});
        else {Array a;for(size_t i=0;i<N;++i)a.push_back(encode(key+"/"+std::to_string(i),x[i]));return Value(a);}
    }
    template<class T,size_t N> void decode(const Value &v,T (&x)[N]) {
        if constexpr(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>){auto a=numeric<T>(v);require(a.size()==N,"fixed array dimensions mismatch");std::copy(a.begin(),a.end(),x);}
        else {const auto &a=array(v);require(a.size()==N,"fixed array dimensions mismatch");for(size_t i=0;i<N;++i)decode(a[i],x[i]);}
    }
    template<class T,size_t N> Value encode(const std::string &key,const std::array<T,N> &x) {
        Array a;for(size_t i=0;i<N;++i)a.push_back(encode(key+"/"+std::to_string(i),x[i]));return Value(a);
    }
    template<class T,size_t N> void decode(const Value &v,std::array<T,N> &x) {
        const auto &a=array(v);require(a.size()==N,"fixed array dimensions mismatch");for(size_t i=0;i<N;++i)decode(a[i],x[i]);
    }
    template<class T> Value encode(const std::string &key,const std::map<std::string,T> &x) {
        Object o;for(const auto &p:x)o[p.first]=encode(key+"/"+p.first,p.second);return Value(o);
    }
    template<class T> void decode(const Value &v,std::map<std::string,T> &x) {
        x.clear();for(const auto &p:object(v))decode(p.second,x[p.first]);
    }
public:
    State(Writer &w,std::string prefix,NumericPool *pool=nullptr):writer_(&w),pool_(pool),prefix_(std::move(prefix)){}
    State(const Reader &r,Object fields,BlockCache *cache=nullptr):cache_(cache),reader_(&r),fields_(std::move(fields)){}
    bool reading()const{return reader_!=nullptr;}
    const Object &fields()const{return fields_;}
    template<class T> void field(const std::string &name,T &value) {
        if(reader_)decode(at(fields_,name),value);else fields_[name]=encode(name,value);
    }
};
} // namespace GPM
