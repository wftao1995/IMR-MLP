#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <cstring>
#include <limits>
#include <sstream>

namespace {
std::string values() { std::ostringstream out; for (int i=0;i<36;++i) { if (i) out<<' '; out<<i-17; } return out.str(); }
// Independent existing BIN grammar: native scalar storage, no codec helper calls.
template<class T> void raw(std::ostream &out,const T &value) { out.write(reinterpret_cast<const char*>(&value),sizeof(value)); }
void string(std::ostream &out,const std::string &value) { int32_t n=value.size(); raw(out,n); out.write(value.data(),n); }
std::string fixture(const std::string &format,const std::string &payload,bool duplicate=false) {
    std::ostringstream out;
    if (format=="cfg") {
        out<<"BEGIN_CFG\nSize\n2\nSupercell\n8 0 0\n0 9 0\n0 0 10\nAtomData: id type cartes_x cartes_y cartes_z\n1 1 1 2 3\n2 8 4 5 6\nFeature hessian "<<payload<<'\n';
        if (duplicate) out<<"Feature hessian "<<payload<<'\n';
        out<<"END_CFG\n";
    } else {
        out.write("GPMLBIN2",8); raw(out,uint32_t(3)); raw(out,int32_t(2));
        out.put('l'); const double cell[9]={8,0,0,0,9,0,0,0,10}; for (double x:cell) raw(out,x);
        out.put('p'); const double pos[6]={1,2,3,4,5,6}; for (double x:pos) raw(out,x);
        out.put('t'); raw(out,int32_t(1)); raw(out,int32_t(8));
        out.put('F'); string(out,"hessian"); string(out,payload);
        if (duplicate) { out.put('F'); string(out,"hessian"); string(out,payload); }
        out.put('#');
    }
    return out.str();
}
Atoms sample() {
    Atoms a; a.pos_={{1,2,3},{4,5,6}}; a.chemmap_={1,8}; a.cell_=8*Eigen::Matrix3d::Identity();
    a.hessian_.resize(6,6); for (int i=0;i<6;++i) for (int j=0;j<6;++j) a.hessian_(i,j)=6*i+j-17;
    a.have_hessian=true; return a;
}
}
TEST_CASE("CFG BIN full Hessian reads independent existing Feature records in row order", "[gpio][hessian][cfg][binary]") {
    const std::string format=GENERATE("cfg","bin");
    std::istringstream input(fixture(format,values())); GPIO io; Atoms restored;
    REQUIRE(io.read_one(restored,input,format)); REQUIRE(restored.have_hessian);
    for (int i=0;i<6;++i) for (int j=0;j<6;++j) REQUIRE(restored.hessian_(i,j)==6*i+j-17);
    REQUIRE(restored.features_.count("hessian")==0); REQUIRE(restored.hessian_(0,1)!=restored.hessian_(1,0));
}
TEST_CASE("CFG BIN full Hessian output reuses Feature without changing BIN header", "[gpio][hessian][cfg][binary]") {
    const std::string format=GENERATE("cfg","bin"); auto a=sample();
    a.features_["hessian"]="old arbitrary metadata";
    a.hvp_directions_=Eigen::MatrixXd::Ones(6,1); a.hvp_targets_=3*a.hvp_directions_;
    GPIO io; std::ostringstream output; REQUIRE(io.write_one(a,output,format));
    if (format=="cfg") REQUIRE(output.str().find("Feature hessian "+values()+"\n")!=std::string::npos);
    else {
        REQUIRE(output.str().starts_with("GPMLBIN2")); uint32_t version; std::memcpy(&version,output.str().data()+8,4); REQUIRE(version==3);
        const auto at=output.str().find("hessian"); REQUIRE(at!=std::string::npos);
        int32_t size; std::memcpy(&size,output.str().data()+at+7,4); REQUIRE(size==values().size());
        REQUIRE(output.str().substr(at+11,size)==values());
    }
    REQUIRE(output.str().find("old arbitrary metadata")==std::string::npos);
    std::istringstream input(output.str()); Atoms restored; REQUIRE(io.read_one(restored,input,format));
    REQUIRE(restored.hessian_==a.hessian_); REQUIRE(restored.hvp_targets_==a.hvp_targets_);
    REQUIRE(a.features_.at("hessian")=="old arbitrary metadata");
}
TEST_CASE("CFG BIN preserve arbitrary legacy Hessian Feature text and reject duplicate matrices", "[gpio][hessian][cfg][binary]") {
    const std::string format=GENERATE("cfg","bin"); GPIO io;
    const std::string payload=GENERATE("user note: acoustic", "1 2 3", "nan 0 0 0 1 0 0 0 1");
    std::istringstream input(fixture(format,payload)); Atoms a; REQUIRE(io.read_one(a,input,format));
    REQUIRE_FALSE(a.have_hessian); REQUIRE(a.features_.at("hessian")==payload);
    std::ostringstream output; REQUIRE(io.write_one(a,output,format));
    REQUIRE(output.str().find(payload)!=std::string::npos);
    std::istringstream duplicate(fixture(format,values(),true)); REQUIRE_THROWS(io.read_one(a,duplicate,format));
}
TEST_CASE("CFG BIN writer rejects invalid physical matrices before frame bytes", "[gpio][hessian][cfg][binary]") {
    const std::string format=GENERATE("cfg","bin"); const int kind=GENERATE(0,1,2); auto a=sample();
    if (kind==0) a.hessian_.resize(5,5);
    if (kind==1) a.hessian_(0,0)=std::numeric_limits<double>::quiet_NaN();
    if (kind==2) a.hessian_(0,0)=std::numeric_limits<double>::infinity();
    a.chemmap_={0,0}; GPIO io; io.setmap({"O"}); io.absolute_index_write=true;
    std::ostringstream output; REQUIRE_THROWS(io.write_one(a,output,format)); REQUIRE(output.str().empty());
    REQUIRE(a.chemmap_==std::vector<int>{0,0});
}
TEST_CASE("BIN dense Hessian respects existing Feature size cap before header", "[gpio][hessian][binary]") {
    Atoms a; a.pos_.resize(100,Eigen::RowVector3d::Zero()); a.chemmap_.resize(100,0);
    a.hessian_=Eigen::MatrixXd::Constant(300,300,1.2345678901234567); a.have_hessian=true;
    GPIO io; io.setmap({"O"}); io.absolute_index_write=true;
    std::ostringstream output; REQUIRE_FALSE(io.write_one(a,output,"bin")); REQUIRE(output.str().empty());
    REQUIRE(a.chemmap_==std::vector<int>(100,0));
}
TEST_CASE("CFG BIN bounded Hessian reader owns frames and reports a late duplicate", "[gpio][hessian][cfg][binary][stream]") {
    const std::string format=GENERATE("cfg","bin");
    std::istringstream input(fixture(format,values())+fixture(format,values(),true)); GPIO io;
    REQUIRE(io.seq_read_init(input,format)); Atoms first; REQUIRE(io.seq_read(input,first));
    REQUIRE(first.hessian_(5,0)==13); Atoms next; REQUIRE_THROWS(io.seq_read(input,next));
    REQUIRE(first.hessian_(5,0)==13); REQUIRE_FALSE(io.seq_read(input,next));
}
