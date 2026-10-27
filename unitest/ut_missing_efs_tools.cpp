// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "apps/fastfit_errors.h"
#include "io/gpio.h"
#include "tools/constants.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>

using Catch::Approx;
namespace {
struct Workspace {
    std::filesystem::path original=std::filesystem::current_path(), path;
    CMDOption::CommandLineOptions saved=CMDOption::global_cmdopts;
    Workspace() {
        char name[]="/tmp/imrmlp_missing_efs_XXXXXX";
        const char* result=mkdtemp(name);
        if (!result) throw std::runtime_error("cannot create test directory");
        path=result;
        std::filesystem::current_path(path);
    }
    ~Workspace() {
        CMDOption::global_cmdopts=saved;
        SetStreamForOutput(&std::cout);
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
};
Atoms frame(int n, int labels) {
    Atoms a;
    a.cell_=5.*Eigen::Matrix3d::Identity();
    a.pos_.assign(n,Eigen::Vector3d::Zero()); a.chemmap_.assign(n,0);
    a.have_energy=labels&1; a.energy_=2.;
    if(labels&2) a.forces_.assign(n,Eigen::Vector3d(1.,0.,0.));
    a.have_stress=labels&4; a.stress_=Eigen::Matrix3d::Zero(); a.stress_(0,0)=1.;
    return a;
}
void write_frames(const std::string& filename, std::vector<Atoms> data) {
    GPIO io; REQUIRE(io.write(data,filename,"mlpcfg"));
}
std::string contents(const std::string& filename) {
    std::ifstream in(filename);
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
int run(int (*command)(int,char**), std::vector<std::string> args) {
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions();
    std::vector<char*> argv;
    for(auto& arg:args) argv.push_back(arg.data());
    return command(argv.size(),argv.data());
}
int compare(std::vector<std::string> options={"--percentile","0.95"}) {
    std::vector<std::string> args={"calc_err","--input","ref.cfg","--input-predict","pred.cfg",
        "--input-format","mlpcfg","--output-summary","summary.txt",
        "--output-energy-diff","energy.dat","--output-energy-at-diff","energy-at.dat",
        "--output-force-diff","force.dat","--output-stress-diff","stress.dat",
        "--output-virial-diff","virial.dat"};
    args.insert(args.end(),options.begin(),options.end());
    return run(calc_err,std::move(args));
}
size_t rows(const std::string& filename) {
    std::istringstream in(contents(filename));
    std::string line; size_t n=0;
    while(std::getline(in,line)) if(!line.empty() && line[0]!='#') ++n;
    return n;
}
}

TEST_CASE("calc_err uses paired labels for statistics and all exports", "[apps][missing-labels]") {
    Workspace workspace;
    auto ref=frame(1,7), zero=ref;
    zero.energy_=0.; zero.forces_[0].setZero(); zero.stress_.setZero();
    write_frames("ref.cfg",{ref,frame(3,0)});
    write_frames("pred.cfg",{zero,frame(3,7)});
    REQUIRE(compare()==0);
    auto report=contents("summary.txt");
    REQUIRE(report.find("Average absolute difference = 2.000000")!=std::string::npos);
    REQUIRE(report.find("RMS     absolute difference = 2.000000")!=std::string::npos);
    REQUIRE(report.find("Errors checked for 1 atoms")!=std::string::npos);
    REQUIRE(report.find("Errors checked for 1 configurations")!=std::string::npos);
    REQUIRE(rows("energy.dat")==1); REQUIRE(rows("energy-at.dat")==1);
    REQUIRE(rows("force.dat")==1); REQUIRE(rows("force.datcmpt")==3);
    REQUIRE(rows("stress.dat")==6); REQUIRE(rows("virial.dat")==6);

    // Reverse the missing side: prediction labels must also be checked.
    write_frames("ref.cfg",{ref,frame(3,7)});
    write_frames("pred.cfg",{zero,frame(3,0)});
    REQUIRE(compare()==0);
    REQUIRE(contents("summary.txt")==report);
    REQUIRE(rows("force.dat")==1);

    // No overlap for any property, despite labels existing on either side.
    write_frames("ref.cfg",{frame(1,1)});
    write_frames("pred.cfg",{frame(1,6)});
    REQUIRE(compare()==0);
    report=contents("summary.txt");
    REQUIRE(report.find("N/A")!=std::string::npos);
    REQUIRE(report.find("nan")==std::string::npos);
    for(const auto* file:{"energy.dat","energy-at.dat","force.dat","force.datcmpt","stress.dat","virial.dat"})
        REQUIRE(rows(file)==0);

    write_frames("ref.cfg",{zero}); write_frames("pred.cfg",{ref});
    REQUIRE(compare()==0);
    report=contents("summary.txt");
    REQUIRE(report.find("Max(ForceDiff) / Max(Force) = N/A")!=std::string::npos);
    REQUIRE(report.find("inf")==std::string::npos);
}

TEST_CASE("calc_err stress component RMS uses the rotation invariant full tensor", "[apps][stress]") {
    Workspace workspace;
    auto ref=frame(1,4), prediction=ref;
    prediction.stress_.setZero();
    const auto metric=[](const std::string& report,const std::string& section,const std::string& name) {
        const auto start=report.find(section);REQUIRE(start!=std::string::npos);
        const auto pos=report.find(name+" = ",start);REQUIRE(pos!=std::string::npos);
        return std::stod(report.substr(pos+name.size()+3));
    };
    for(const double angle:{0.,M_PI/4.}) {
        const Eigen::Matrix3d rotation=Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitZ()).toRotationMatrix();
        auto rotated=ref;
        rotated.stress_=(rotation*ref.stress_*rotation.transpose()).eval();
        // Unlabelled frames must not enter the RMSE denominator.
        write_frames("ref.cfg",{rotated,frame(1,0)});
        write_frames("pred.cfg",{prediction,frame(1,0)});
        REQUIRE(compare()==0);
        const auto report=contents("summary.txt");
        REQUIRE(metric(report,"Stress components (eV):","RMS     absolute difference")
            ==Approx(1./3.).margin(5e-7));
        REQUIRE(metric(report,"Virial components (GPa):","RMS     absolute difference")
            ==Approx(eVA3_to_GPa/ref.get_volume()/3.).margin(5e-7));
        REQUIRE(metric(report,"Stress components (eV):","RMS(StressDiff)/RMS(Stress)")
            ==Approx(1.));
        REQUIRE(metric(report,"Virial components (GPa):","RMS(VirialDiff)/RMS(Virial)")
            ==Approx(1.));
        const double expected_mae=rotated.stress_.cwiseAbs().sum()/9.;
        REQUIRE(metric(report,"Stress components (eV):","Average absolute difference")
            ==Approx(expected_mae).margin(5e-7));
    }
}

TEST_CASE("calc_err rejects mismatched and empty inputs", "[apps][missing-labels]") {
    Workspace workspace;
    write_frames("ref.cfg",{frame(1,7)});
    SECTION("frame count") { write_frames("pred.cfg",{frame(1,7),frame(1,7)}); }
    SECTION("atom count") { write_frames("pred.cfg",{frame(2,7)}); }
    SECTION("empty prediction") { std::ofstream("pred.cfg"); }
    SECTION("missing prediction") {}
    REQUIRE_THROWS_AS(compare(),GPException);
}

TEST_CASE("calc_err weights force statistics by labelled atoms", "[apps][missing-labels]") {
    Workspace workspace;
    auto one=frame(1,7), three=frame(3,7);
    for(auto& f:three.forces_) f*=3.;
    auto p1=one,p3=three;
    p1.energy_=p3.energy_=0.; p1.stress_.setZero(); p3.stress_.setZero();
    for(auto* a:{&p1,&p3}) for(auto& f:a->forces_) f.setZero();
    write_frames("ref.cfg",{one,three,frame(2,0)});
    write_frames("pred.cfg",{p1,p3,frame(2,7)});
    REQUIRE(compare()==0);
    const auto report=contents("summary.txt");
    REQUIRE(report.find("Errors checked for 4 atoms")!=std::string::npos);
    REQUIRE(report.find("Average absolute difference = 2.500000")!=std::string::npos);
    REQUIRE(report.find("RMS     absolute difference = 2.645751")!=std::string::npos);
    REQUIRE(report.find("P95    absolute difference = 3.000000")!=std::string::npos);
    REQUIRE(rows("force.dat")==4);
}

TEST_CASE("calc_err reports valid percentile endpoints and restores its output stream", "[apps][calc-err]") {
    Workspace workspace;
    const std::string percentile=GENERATE("0","0.95","1");
    auto first=frame(1,1), second=first, prediction=first;
    first.energy_=1.; second.energy_=3.; prediction.energy_=0.;
    write_frames("ref.cfg",{first,second});
    write_frames("pred.cfg",{prediction,prediction});
    auto *capture=new std::ostringstream;
    ScopedStreamRedirect capture_guard(global_ostream,capture);
    REQUIRE(compare({"--percentile",percentile})==0);
    REQUIRE(global_ostream==capture);
    REQUIRE(capture->str().find("Complete!")!=std::string::npos);
    const auto report=contents("summary.txt");
    if(percentile=="0") {
        REQUIRE(report.find("\tP")==std::string::npos);
    } else {
        const std::string expected=percentile=="1" ? "P100    absolute difference = 3.000000"
                                                 : "P95    absolute difference = 2.900000";
        REQUIRE(report.find(expected)!=std::string::npos);
    }
}

TEST_CASE("calc_err rejects invalid percentiles before reading input", "[apps][calc-err]") {
    Workspace workspace;
    const std::string percentile=GENERATE("-0.1","1.1","nan","inf");
    REQUIRE_THROWS_WITH(compare({"--percentile",percentile}),
        Catch::Matchers::ContainsSubstring("percentile"));
    REQUIRE_FALSE(std::filesystem::exists("summary.txt"));
}

TEST_CASE("calc_err validates the effective percentile after settings", "[apps][calc-err]") {
    Workspace workspace;
    std::ofstream("report.ini") << "percentile 1.1\n";
    SECTION("invalid settings") {
        REQUIRE_THROWS_WITH(compare({"--settings","report.ini"}),
            Catch::Matchers::ContainsSubstring("percentile"));
        REQUIRE_FALSE(std::filesystem::exists("summary.txt"));
    }
    SECTION("explicit CLI value overrides settings") {
        write_frames("ref.cfg",{frame(1,1)});
        write_frames("pred.cfg",{frame(1,1)});
        REQUIRE(compare({"--settings","report.ini","--percentile","1"})==0);
        REQUIRE(contents("summary.txt").find("P100")!=std::string::npos);
    }
}

TEST_CASE("calc_err reports output open failures and restores its output stream", "[apps][calc-err]") {
    Workspace workspace;
    write_frames("ref.cfg",{frame(1,7)});
    write_frames("pred.cfg",{frame(1,7)});
    const std::string blocked=GENERATE("summary.txt","energy.dat","force.datcmpt");
    std::filesystem::create_directory(blocked);
    auto *capture=new std::ostringstream;
    ScopedStreamRedirect capture_guard(global_ostream,capture);
    REQUIRE_THROWS_WITH(compare(),Catch::Matchers::ContainsSubstring("failed to open output file: "+blocked));
    REQUIRE(global_ostream==capture);
    Message("after failure");
    REQUIRE(capture->str().find("after failure")!=std::string::npos);
    REQUIRE(capture->str().find("Complete!")==std::string::npos);
}

TEST_CASE("calc_err detects a failed summary write and restores its output stream", "[apps][calc-err]") {
    if(!std::filesystem::exists("/dev/full")) SKIP("requires /dev/full");
    Workspace workspace;
    write_frames("ref.cfg",{frame(1,7)});
    write_frames("pred.cfg",{frame(1,7)});
    std::filesystem::create_symlink("/dev/full","summary.txt");
    auto *capture=new std::ostringstream;
    ScopedStreamRedirect capture_guard(global_ostream,capture);
    REQUIRE_THROWS_WITH(compare(),Catch::Matchers::ContainsSubstring("failed to write output file: summary.txt"));
    REQUIRE(global_ostream==capture);
    Message("after write failure");
    REQUIRE(capture->str().find("after write failure")!=std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists("energy.dat"));
}

TEST_CASE("calc_ferr requires complete paired energies", "[apps][missing-labels]") {
    Workspace workspace;
    write_frames("ref.cfg",{frame(1,1)}); write_frames("mlp.cfg",{frame(1,1)});
    std::string temperature="300";
    bool valid=false;
    SECTION("complete energies") { valid=true; }
    SECTION("reference energy absent") { write_frames("ref.cfg",{frame(1,0)}); }
    SECTION("prediction energy absent") { write_frames("mlp.cfg",{frame(1,0)}); }
    SECTION("frame count") { write_frames("mlp.cfg",{frame(1,1),frame(1,1)}); }
    SECTION("atom count") { write_frames("mlp.cfg",{frame(2,1)}); }
    SECTION("empty reference") { std::ofstream("ref.cfg"); }
    SECTION("temperature") { temperature="0"; }
    auto call=[&] { return run(calc_ferr,{"calc_ferr","--input","ref.cfg","--input-format","mlpcfg","--temperature",temperature}); };
    if(valid) REQUIRE(call()==0);
    else REQUIRE_THROWS_AS(call(),GPException);
}

TEST_CASE("fastfit accumulates only present EFS labels", "[fastfit][missing-labels]") {
    func_fastfit::ErrorsReport errors(1.,1.,1.);
    auto prediction=frame(1,7); prediction.energy_=0.;
    prediction.forces_[0].setZero(); prediction.stress_.setZero();
    for(int mask=0;mask<8;++mask) {
        auto ref=frame(1,mask); errors.add(prediction,ref);
    }
    double e,f,s,loss;
    errors.Reduce(e,f,s,loss);
    REQUIRE(e==Approx(2.)); REQUIRE(f==Approx(1.)); REQUIRE(s==Approx(1.));
    REQUIRE(loss==Approx(24.));
    errors.clear(); auto missing=frame(1,0); errors.add(prediction,missing);
    REQUIRE_THROWS_AS(errors.Reduce(e,f,s,loss),GPException);
    errors.clear(); // An absent optional validation set is allowed.
    REQUIRE_NOTHROW(errors.Reduce(e,f,s,loss));
    auto invalid=frame(2,2); invalid.forces_.resize(1);
    auto two=frame(2,7);
    REQUIRE_THROWS_AS(errors.add(two,invalid),GPException);
    func_fastfit::ErrorsReport disabled(0.,0.,0.);
    disabled.add(prediction,prediction);
    REQUIRE_THROWS_AS(disabled.Reduce(e,f,s,loss),GPException);
}

TEST_CASE("forced mlpcfg EFS output rejects absent labels before writing", "[gpio][missing-labels]") {
    int mask=GENERATE(0,1,2,3,4,5,6,7);
    GPIO io; io.cfg_have_efs=true;
    Atoms a=frame(1,mask); std::ostringstream out;
    REQUIRE(io.write_one(a,out,"mlpcfg")== (mask==7));
    if(mask!=7) REQUIRE(out.str().empty());
}

TEST_CASE("unsupported formats leave output files untouched", "[gpio][formats]") {
    Workspace workspace;
    const std::string format=GENERATE("", "nc", "netcdf", "unsupported-format");
    std::ofstream("target.nc") << "preserve";
    GPIO io;
    auto atom=frame(1,7);
    std::vector<Atoms> data={atom}, output;
    REQUIRE_FALSE(io.write(data,"target.nc",format));
    REQUIRE(contents("target.nc")=="preserve");
    REQUIRE_FALSE(io.write_one(atom,"target.nc",format));
    REQUIRE(contents("target.nc")=="preserve");
    REQUIRE_FALSE(io.read(output,"target.nc",format));
    REQUIRE_FALSE(io.read_one(atom,"target.nc",format));
    std::istringstream in("preserve");
    REQUIRE_FALSE(io.seq_read_init(in,format));
    std::ostringstream out;
    REQUIRE_FALSE(io.seq_write_init(out,format,"target.nc"));
    REQUIRE(out.str().empty());
}

TEST_CASE("convstr rejects unsupported formats before opening output", "[apps][formats]") {
    Workspace workspace;
    const std::string format=GENERATE("", "nc", "netcdf");
    const bool invalid_input=GENERATE(false,true);
    write_frames("input.cfg",{frame(1,7)});
    write_frames("input.nc",{frame(1,7)});
    std::ofstream("target.nc") << "preserve";
    std::vector<std::string> args={"convstr", invalid_input ? "input.nc" : "input.cfg",
        "target.nc", "--overwrite"};
    if(invalid_input) {
        args.insert(args.end(),{"--output-format","mlpcfg"});
        if(!format.empty()) args.insert(args.end(),{"--input-format",format});
    } else {
        args.insert(args.end(),{"--input-format","mlpcfg"});
        if(!format.empty()) args.insert(args.end(),{"--output-format",format});
    }
    REQUIRE(run(convstr,args)!=0);
    REQUIRE(contents("target.nc")=="preserve");
}

TEST_CASE("convstr rejects unsupported compression before opening output", "[apps][compressed]") {
    Workspace workspace;
    const bool invalid_input=GENERATE(false,true);
    const std::string input=GENERATE("input.xyz.zst", "input.xyz");
    std::ofstream(input,std::ios::binary) << "\x28\xb5\x2f\xfd";
    write_frames("input.cfg",{frame(1,7)});
    const std::string output=invalid_input ? "target.cfg" : "target.cfg.zst";
    std::ofstream(output) << "preserve";
    REQUIRE(run(convstr,{"convstr", invalid_input ? input : "input.cfg", output,
        "--input-format", "mlpcfg", "--output-format", "mlpcfg", "--overwrite"})!=0);
    REQUIRE(contents(output)=="preserve");
}
