#include <catch2/catch_all.hpp>
#include "io/gpio_session.h"
#include <algorithm>
#include <barrier>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace {
struct SessionWorkspace {
    std::filesystem::path path;
    SessionWorkspace() {
        char name[]="/tmp/imrmlp_gpio_sessions_XXXXXX";
        auto created=mkdtemp(name);
        if(!created) throw std::runtime_error("mkdtemp failed");
        path=created;
    }
    ~SessionWorkspace(){std::filesystem::remove_all(path);}
    std::string file(const std::string& name) const {return (path/name).string();}
};
std::string xyz_frame(int frame, const std::string& symbol="H") {
    return "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 energy="+
        std::to_string(frame)+"\n"+symbol+" "+std::to_string(frame)+" 2 3\n";
}
std::string xyz_frames(int count) {
    std::string text;for(int i=0;i<count;++i)text+=xyz_frame(i);return text;
}
std::unique_ptr<std::istream> text_stream(const std::string& text) {
    return std::make_unique<std::istringstream>(text);
}
Atoms session_atom(double scale=10.) {
    Atoms a;a.cell_=scale*Eigen::Matrix3d::Identity();a.pos_={{.1,.2,.3}};
    a.chemmap_={0};a.pbc_={true,true,true};a.cartesian=false;
    a.have_energy=true;a.energy_=scale;a.features_["test"]="preserved";return a;
}
std::vector<Atoms> collect(GPIOReader& reader) {
    std::vector<Atoms> result;while(auto a=reader.next())result.push_back(std::move(*a));return result;
}
std::string xdatcar(double length) {
    return "session\n1\n"+std::to_string(length)+" 0 0\n0 "+std::to_string(length)+
        " 0\n0 0 "+std::to_string(length)+"\nH\n1\n"
        "Direct configuration= 1\n.1 .2 .3\nDirect configuration= 2\n.4 .5 .6\n";
}
// No stream buffering can read past the boundary without recording it.
struct BoundaryState {std::size_t position=0,boundary=0;int forbidden_reads=0;};
struct BoundaryBuffer : std::streambuf {
    std::string text;std::shared_ptr<BoundaryState> state;
    BoundaryBuffer(std::string value,std::shared_ptr<BoundaryState> s):text(std::move(value)),state(std::move(s)){}
    int_type underflow() override {
        if(state->position>=state->boundary){++state->forbidden_reads;throw std::runtime_error("unselected suffix accessed");}
        return traits_type::to_int_type(text.at(state->position));
    }
    int_type uflow() override {auto c=underflow();++state->position;return c;}
};
struct BoundaryInput : std::istream {
    BoundaryBuffer buffer;
    BoundaryInput(std::string text,std::shared_ptr<BoundaryState> state):std::istream(nullptr),buffer(std::move(text),std::move(state)){rdbuf(&buffer);}
};
}

TEST_CASE("Reader sessions snapshot independent ranges and chemistry", "[gpio][sessions]") {
    SessionWorkspace temp;GPIO configuration;
    {std::ofstream map(temp.file("first.map"));map<<"H 0\nHe 0\n";}
    {std::ofstream map(temp.file("second.map"));map<<"He 0\nH 0\n";}
    REQUIRE(configuration.readmap(temp.file("first.map")));
    configuration.read_startfrom=1;configuration.read_endto=4;configuration.read_irate=2;
    GPIOReader first(configuration,text_stream(xyz_frames(6)),"extxyz");
    REQUIRE(configuration.readmap(temp.file("second.map")));
    configuration.read_startfrom=0;configuration.read_endto=2;configuration.read_irate=1;
    GPIOReader second(configuration,text_stream(xyz_frames(6)),"extxyz");
    auto a=first.next(),b=second.next();REQUIRE(a);REQUIRE(b);
    REQUIRE(a->id==1);REQUIRE(a->chemmap_==std::vector<int>{0});
    REQUIRE(b->id==0);REQUIRE(b->chemmap_==std::vector<int>{1});
    auto rest_a=collect(first),rest_b=collect(second);
    REQUIRE(rest_a.size()==1);REQUIRE(rest_a[0].id==3);
    REQUIRE(rest_b.size()==2);REQUIRE(rest_b[0].id==1);REQUIRE(rest_b[1].id==2);
    REQUIRE_FALSE(first.next());REQUIRE_FALSE(second.next());
    REQUIRE(a->pos_[0][0]==1.); // Returned frames remain owned after later reads/close.
}

TEST_CASE("Header sessions interleave without replacing another parser state", "[gpio][sessions][xdatcar]") {
    GPIO configuration;std::istringstream legacy_input(xdatcar(2.));Atoms legacy;
    REQUIRE(configuration.seq_read_init(legacy_input,"xdatcar"));
    REQUIRE(configuration.seq_read(legacy_input,legacy));REQUIRE(legacy.id==0);
    GPIOReader first(configuration,text_stream(xdatcar(4.)),"xdatcar");
    GPIOReader second(configuration,text_stream(xdatcar(9.)),"xdatcar");
    for(int i=0;i<2;++i) {
        auto a=first.next(),b=second.next();REQUIRE(a);REQUIRE(b);
        REQUIRE(a->id==i);REQUIRE(b->id==i);
        REQUIRE(a->cell_(0,0)==4.);REQUIRE(b->cell_(0,0)==9.);
        a->to_fractional();b->to_fractional();
        REQUIRE(a->pos_[0][0]==Catch::Approx(i==0?.1:.4));
        REQUIRE(b->pos_[0][2]==Catch::Approx(i==0?.3:.6));
    }
    REQUIRE_FALSE(first.next());REQUIRE_FALSE(second.next());
    REQUIRE(configuration.seq_read(legacy_input,legacy));
    REQUIRE(legacy.id==1);REQUIRE(legacy.cell_(0,0)==2.);
}

TEST_CASE("Reader close and malformed-input errors remain local", "[gpio][sessions]") {
    GPIO configuration;
    GPIOReader broken(configuration,text_stream(xyz_frame(0)+"2\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"),"extxyz");
    GPIOReader healthy(configuration,text_stream(xyz_frames(3)),"extxyz");
    GPIOReader closed(configuration,text_stream(xyz_frames(3)),"extxyz");
    REQUIRE(broken.next());REQUIRE(healthy.next());closed.close();closed.close();
    REQUIRE_FALSE(closed.next());REQUIRE_THROWS(broken.next());REQUIRE_FALSE(broken.next());
    auto remaining=collect(healthy);REQUIRE(remaining.size()==2);REQUIRE(remaining.back().id==2);
    GPIOReader empty(configuration,text_stream(""),"extxyz");
    REQUIRE_FALSE(empty.next());empty.close();REQUIRE_FALSE(empty.next());
    REQUIRE_THROWS(GPIOReader(configuration,text_stream("bad header\n"),"xdatcar"));
    GPIOReader restarted(configuration,text_stream(xyz_frame(7)),"extxyz");
    auto a=restarted.next();REQUIRE(a);REQUIRE(a->id==0);REQUIRE(a->energy_==7.);
}

TEST_CASE("Reader selected end closes before accessing an unselected bad suffix", "[gpio][sessions][range]") {
    GPIO configuration;configuration.read_endto=3;configuration.read_irate=2;
    const auto prefix=xyz_frames(3);
    auto state=std::make_shared<BoundaryState>();state->boundary=prefix.size();
    auto input=std::make_unique<BoundaryInput>(prefix+"this suffix must stay unread",state);
    GPIOReader reader(configuration,std::move(input),"extxyz");
    auto frames=collect(reader);REQUIRE(frames.size()==2);
    REQUIRE(frames[0].id==0);REQUIRE(frames[1].id==2);
    REQUIRE(state->position==state->boundary);REQUIRE(state->forbidden_reads==0);
    REQUIRE_FALSE(reader.next());REQUIRE(state->forbidden_reads==0);
}

TEST_CASE("Concurrent next calls on one session return each frame once", "[gpio][sessions][threads]") {
    GPIO configuration;GPIOReader reader(configuration,text_stream(xyz_frames(80)),"extxyz");
    std::barrier start(4);std::mutex results_mutex;std::vector<int> ids;std::exception_ptr failure;
    auto worker=[&] {
        start.arrive_and_wait();
        try {while(auto atom=reader.next()){std::lock_guard lock(results_mutex);ids.push_back(atom->id);}}
        catch(...){std::lock_guard lock(results_mutex);failure=std::current_exception();}
    };
    {std::vector<std::jthread> threads;for(int i=0;i<4;++i)threads.emplace_back(worker);}
    if(failure)std::rethrow_exception(failure);
    std::sort(ids.begin(),ids.end());REQUIRE(ids.size()==80);
    for(int i=0;i<80;++i)REQUIRE(ids[i]==i);
    REQUIRE_FALSE(reader.next());
}

TEST_CASE("Independent writer headers preserve fractional input and each other", "[gpio][sessions][xdatcar]") {
    SessionWorkspace temp;GPIO configuration;configuration.setmap(std::vector<std::string>{"H"});
    GPIOWriter first(configuration,temp.file("one.xdatcar"),"xdatcar");
    GPIOWriter second(configuration,temp.file("two.xdatcar"),"xdatcar");
    auto a=session_atom(4.),b=session_atom(9.);const auto original_a=a,original_b=b;
    first.write(a);second.write(b);first.write(a);second.write(b);first.close();second.close();
    REQUIRE_FALSE(a.cartesian);REQUIRE_FALSE(b.cartesian);
    REQUIRE(a.pos_==original_a.pos_);REQUIRE(b.pos_==original_b.pos_);
    REQUIRE(a.cell_==original_a.cell_);REQUIRE(a.features_==original_a.features_);
    GPIOReader r1(configuration,temp.file("one.xdatcar"),"xdatcar");
    GPIOReader r2(configuration,temp.file("two.xdatcar"),"xdatcar");
    auto one=collect(r1),two=collect(r2);REQUIRE(one.size()==2);REQUIRE(two.size()==2);
    for(auto& frame:one){REQUIRE(frame.cell_(0,0)==4.);frame.to_fractional();REQUIRE(frame.pos_[0].isApprox(original_a.pos_[0]));}
    for(auto& frame:two){REQUIRE(frame.cell_(0,0)==9.);frame.to_fractional();REQUIRE(frame.pos_[0].isApprox(original_b.pos_[0]));}
    REQUIRE_THROWS_AS(first.write(a),std::logic_error);first.close();
}

TEST_CASE("Writer range counters are independent and empty close preserves existing paths", "[gpio][sessions][range]") {
    SessionWorkspace temp;GPIO configuration;configuration.setmap(std::vector<std::string>{"H"});configuration.out_startfrom=1;configuration.out_endto=3;configuration.outrate=2;
    GPIOWriter first(configuration,temp.file("one.xyz"),"extxyz");
    GPIOWriter second(configuration,temp.file("two.xyz"),"extxyz");
    auto a=session_atom();
    for(int i=0;i<5;++i){a.energy_=i;first.write(a);a.energy_=100+i;second.write(a);}
    first.close();second.close();GPIO reader_configuration;
    GPIOReader r1(reader_configuration,temp.file("one.xyz"),"extxyz"),r2(reader_configuration,temp.file("two.xyz"),"extxyz");
    auto one=collect(r1),two=collect(r2);REQUIRE(one.size()==2);REQUIRE(two.size()==2);
    REQUIRE(one[0].energy_==1.);REQUIRE(one[1].energy_==3.);
    REQUIRE(two[0].energy_==101.);REQUIRE(two[1].energy_==103.);
    {std::ofstream output(temp.file("empty.xyz"));output<<"existing contents\n";}
    {GPIOWriter empty(configuration,temp.file("empty.xyz"),"extxyz");empty.close();empty.close();REQUIRE_THROWS_AS(empty.write(a),std::logic_error);}
    std::ifstream preserved(temp.file("empty.xyz"));std::string text((std::istreambuf_iterator<char>(preserved)),{});
    REQUIRE(text=="existing contents\n");
    {GPIOWriter empty(configuration,temp.file("absent.xyz"),"extxyz");}
    REQUIRE_FALSE(std::filesystem::exists(temp.file("absent.xyz")));
}

TEST_CASE("Failed writer closes locally and reader configuration bounds are validated", "[gpio][sessions]") {
    SessionWorkspace temp;GPIO configuration;auto a=session_atom();
    GPIOWriter failed(configuration,temp.file("missing/output.xyz"),"extxyz");
    GPIOWriter good(configuration,temp.file("healthy.xyz"),"extxyz");
    REQUIRE_THROWS(failed.write(a));REQUIRE_THROWS_AS(failed.write(a),std::logic_error);
    REQUIRE_NOTHROW(good.write(a));REQUIRE_NOTHROW(good.close());
    GPIOReader healthy(configuration,temp.file("healthy.xyz"),"extxyz");REQUIRE(healthy.next());
    configuration.read_irate=0;
    REQUIRE_THROWS_AS(GPIOReader(configuration,text_stream(xyz_frames(2)),"extxyz"),std::invalid_argument);
    configuration.read_irate=1;configuration.read_startfrom=2;configuration.read_endto=1;
    REQUIRE_THROWS_AS(GPIOReader(configuration,text_stream(xyz_frames(2)),"extxyz"),std::invalid_argument);
    configuration.outrate=0;
    REQUIRE_THROWS_AS(GPIOWriter(configuration,temp.file("bad-range.xyz"),"extxyz"),std::invalid_argument);
    GPIO normal;REQUIRE_THROWS_AS(GPIOReader(normal,text_stream(xyz_frame(0)),"unsupported"),std::invalid_argument);
    REQUIRE_THROWS_AS(GPIOWriter(normal,temp.file("unsupported"),"xsf"),std::invalid_argument);
}

TEST_CASE("Independent sessions advance in parallel with separate snapshots", "[gpio][sessions][threads]") {
    GPIO configuration;configuration.read_startfrom=0;configuration.read_endto=39;configuration.read_irate=2;
    GPIOReader first(configuration,text_stream(xyz_frames(50)),"extxyz");
    configuration.read_startfrom=1;configuration.read_endto=40;configuration.read_irate=3;
    GPIOReader second(configuration,text_stream(xyz_frames(50)),"extxyz");
    std::barrier start(2);std::vector<Atoms> left,right;std::exception_ptr failures[2];
    auto worker=[&](GPIOReader& reader,std::vector<Atoms>& frames,int slot) {
        start.arrive_and_wait();try{frames=collect(reader);}catch(...){failures[slot]=std::current_exception();}
    };
    {
        std::jthread a(worker,std::ref(first),std::ref(left),0);
        std::jthread b(worker,std::ref(second),std::ref(right),1);
    }
    for(auto failure:failures)if(failure)std::rethrow_exception(failure);
    REQUIRE(left.size()==20);REQUIRE(right.size()==14);
    for(size_t i=0;i<left.size();++i){REQUIRE(left[i].id==2*i);REQUIRE(left[i].energy_==2*i);}
    for(size_t i=0;i<right.size();++i){REQUIRE(right[i].id==1+3*i);REQUIRE(right[i].energy_==1+3*i);}
}

TEST_CASE("Session metadata filters snapshot global and local tag configuration", "[gpio][sessions][metadata]") {
    GPIO parent;parent.read_all_keys=false;parent.gtags_={"keep"};parent.tags_={"local_keep"};
    auto text=xyz_frame(0)+xyz_frame(1);
    for(const auto& energy:{std::string("energy=0"),std::string("energy=1")}) {
        const auto at=text.find(energy);text.insert(at+energy.size()," keep=11 drop=22");
    }
    GPIOReader first(parent,text_stream(text),"extxyz");
    // tags_ has no filtering effect in the current EXTXYZ decoder. Verify its
    // clone contract directly without exercising a protected codec.
    auto snapshot=parent.new_session_state();
    parent.gtags_={"drop"};parent.tags_={"local_drop"};
    GPIOReader second(parent,text_stream(text),"extxyz");
    parent.read_all_keys=true;parent.gtags_.clear();parent.tags_.clear();
    REQUIRE_FALSE(snapshot->read_all_keys);
    REQUIRE(snapshot->gtags_==std::set<std::string>{"keep"});
    REQUIRE(snapshot->tags_==std::set<std::string>{"local_keep"});
    for(int i=0;i<2;++i) {
        auto a=first.next(),b=second.next();REQUIRE(a);REQUIRE(b);
        REQUIRE(a->global_attribs_d_.at("keep")==11.);
        REQUIRE(a->global_attribs_d_.count("drop")==0);
        REQUIRE(b->global_attribs_d_.at("drop")==22.);
        REQUIRE(b->global_attribs_d_.count("keep")==0);
        REQUIRE(a->has_energy());REQUIRE(b->has_energy());
        REQUIRE(a->energy_==i);REQUIRE(b->energy_==i);
    }
    REQUIRE_FALSE(first.next());REQUIRE_FALSE(second.next());
    GPIOReader unfiltered(parent,text_stream(text),"extxyz");
    auto a=unfiltered.next();REQUIRE(a);
    REQUIRE(a->global_attribs_d_.at("keep")==11.);REQUIRE(a->global_attribs_d_.at("drop")==22.);
}
