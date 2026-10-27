#pragma once
#include "tools/gpm_container.h"
#include <Eigen/Core>
#include <type_traits>

namespace GPM {
// Named state fields. Numeric arrays have a portable row-major representation;
// this is also used by AL and configuration records, never by object memcpy.
class State {
    Writer *writer_=nullptr;
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
        std::string id=prefix_+"/"+key;writer_->add(id,dtype<T>(),std::move(dims),std::move(bytes));
        return Value(Object{{"block",Value(id)}});
    }
    template<class T> std::vector<T> numeric(const Value &value,std::vector<uint64_t> *dims=nullptr) {
        std::string id=string(at(object(value),"block"));const auto &b=reader_->info(id);
        require(b.dtype==dtype<T>(),"state field dtype mismatch: "+id);if(dims)*dims=b.shape;
        Bytes bytes=reader_->read(id);std::vector<T> result(bytes.size()/sizeof(T));
        for(size_t i=0;i<result.size();++i){uint64_t bits=get_le(bytes.data()+i*sizeof(T),sizeof(T));
            if constexpr(std::is_same_v<T,double>)std::memcpy(&result[i],&bits,8);
            else if constexpr(std::is_signed_v<T>){using U=std::make_unsigned_t<T>;U x=static_cast<U>(bits);std::memcpy(&result[i],&x,sizeof(T));}
            else result[i]=static_cast<T>(bits);}
        return result;
    }
    template<class T> Value encode(const std::string &key,const T &x) {
        if constexpr(std::is_same_v<T,std::string> || std::is_same_v<T,bool>)return Value(x);
        else if constexpr(std::is_same_v<T,double>)return numeric(key,&x,1,{1});
        else if constexpr(std::is_enum_v<T>)return Value(double(static_cast<int>(x)));
        else if constexpr(std::is_integral_v<T> && sizeof(T)<8)return Value(double(x));
        else if constexpr(std::is_integral_v<T>)return Value(std::to_string(x));
        else static_assert(sizeof(T)==0,"unsupported GPM state field");
    }
    template<class T> void decode(const Value &v,T &x) {
        if constexpr(std::is_same_v<T,std::string>)x=string(v);
        else if constexpr(std::is_same_v<T,bool>){require(v.is<bool>(),"expected bool");x=v.get<bool>();}
        else if constexpr(std::is_same_v<T,double>){auto a=numeric<double>(v);require(a.size()==1,"expected scalar");x=a[0];}
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
    State(Writer &w,std::string prefix):writer_(&w),prefix_(std::move(prefix)){}
    State(const Reader &r,Object fields):reader_(&r),fields_(std::move(fields)){}
    bool reading()const{return reader_!=nullptr;}
    const Object &fields()const{return fields_;}
    template<class T> void field(const std::string &name,T &value) {
        if(reader_)decode(at(fields_,name),value);else fields_[name]=encode(name,value);
    }
};
} // namespace GPM
