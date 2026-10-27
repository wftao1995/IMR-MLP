// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "potentials/potential_mtp.h"
#include "mindriver/relax_new.h"
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "apps/modefollowing.h"

namespace mf = AppsModeFollowing;

namespace {
struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    std::filesystem::path original = std::filesystem::current_path(), path;
    int threads = omp_num_threads;
    Workspace() {
        char name[] = "/tmp/imrmlp_modefollowing_audit_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("mkdtemp failed");
        path = dir;
        std::filesystem::current_path(path);
        omp_num_threads = 1; global_omp_init();
        std::ofstream("index.map") << "H\n";
        // For an isolated pair, E(r) = (r-5)^2 (r-1), r < 5.
        // Exact radial saddle: r=7/3, E=256/27, radial eigenvalue=-16.
        std::ofstream("pair.mtp") << R"(MTP
version = 1.1.0
potential_name = analytic_pair
scaling = 1
species_count = 1
potential_tag =
radial_basis_type = RBChebyshev
min_dist = 1
max_dist = 5
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{1, 1}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0, 0, 0, 0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0}
moment_coeffs = {1}
)";
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads; global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        CMDOption::global_cmdopts.set_helponexit(false);
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return modefollowing(argv.size(), argv.data());
    }
};

Atoms structure(int pairs = 1) {
    Atoms a;
    a.cell_ << 30, 1, 2, 0, 32, 3, 0, 0, 34;
    for (int k=0; k<pairs; ++k) {
        Eigen::RowVector3d p = Eigen::RowVector3d::Constant(2+8*k);
        a.pos_.push_back(p);
        a.pos_.push_back(p + Eigen::RowVector3d(2.2, 0.1, 0.2));
        a.chemmap_.insert(a.chemmap_.end(), {0, 0});
    }
    return a;
}

double analytic_energy(const Atoms &a) {
    double energy = 0;
    for (int i=0; i<a.size(); ++i)
        for (int j=0; j<i; ++j) {
            const double r = (a.pos_[i]-a.pos_[j]).norm();
            if (r<5) energy += (r-5)*(r-5)*(r-1);
        }
    return energy;
}

void hessian(Atoms &a, Potentials::PotMTP &pot, Eigen::Matrix3d &ref,
             Eigen::VectorXd &values, Eigen::MatrixXd &vectors, Eigen::MatrixXd &h) {
    mf::compute_hessian(a, ref, 0.0, pot, 1e-7, values, vectors, h);
}
}

TEST_CASE("Analytic MTP fixture agrees with independent pair energy and force", "[modefollowing][fixture]") {
    Workspace work;
    Potentials::PotMTP pot("pair.mtp");
    auto a = structure(3);
    pot.getEFSOMP(a);
    CHECK(a.energy_ == Catch::Approx(analytic_energy(a)).margin(1e-10));
    for (int i=0; i<a.size(); ++i)
        for (int j=0; j<3; ++j) {
            auto plus=a, minus=a;
            plus.pos_[i](j)+=1e-5; minus.pos_[i](j)-=1e-5;
            CHECK(-a.forces_[i](j) == Catch::Approx((analytic_energy(plus)-analytic_energy(minus))/2e-5).margin(1e-8));
        }
}

TEST_CASE("Hessian matches independent energy finite differences", "[modefollowing][hessian]") {
    Workspace work;
    const bool fixed = GENERATE(true, false);
    const bool cartesian = GENERATE(true, false);
    mf::nocell = fixed; mf::car_rep = cartesian;
    mf::offset = fixed ? 0 : (cartesian ? 9 : 6);
    const int pairs = GENERATE(1, 3);
    auto a = structure(pairs);
    Eigen::Matrix3d ref = a.cell_, invref = ref.inverse();
    Eigen::VectorXd values; Eigen::MatrixXd vectors, h;
    Potentials::PotMTP pot("pair.mtp");
    hessian(a, pot, ref, values, vectors, h);
    Eigen::RowVectorXd x;
    if (cartesian) mf::LatPos2X(a, x); else mf::LatPos2x(a, invref, x);
    const auto energy = [&](Eigen::RowVectorXd q) {
        auto trial = a;
        if (cartesian) mf::X2LatPos(trial, q); else mf::x2LatPos(trial, ref, q);
        return analytic_energy(trial);
    };
    for (double delta : {1e-5, 5e-6}) {
        Eigen::MatrixXd fd(x.size(), x.size());
        for (int i=0; i<x.size(); ++i)
            for (int j=0; j<x.size(); ++j) {
                auto pp=x, pm=x, mp=x, mm=x;
                pp(i)+=delta; pp(j)+=delta; pm(i)+=delta; pm(j)-=delta;
                mp(i)-=delta; mp(j)+=delta; mm(i)-=delta; mm(j)-=delta;
                fd(i,j)=(energy(pp)-energy(pm)-energy(mp)+energy(mm))/(4*delta*delta);
            }
        Eigen::Index row, col;
        double error=(h-fd).cwiseAbs().maxCoeff(&row, &col);
        std::cout << "AUDIT HESSIAN fixed=" << fixed << " cartesian=" << cartesian
                  << " delta=" << delta << " max_error=" << error << " at " << row << ',' << col
                  << " analytic=" << h(row,col) << " finite_difference=" << fd(row,col) << '\n';
        INFO("fixed=" << fixed << " cartesian=" << cartesian << " delta=" << delta);
        CHECK(error < 0.003);
    }
}

TEST_CASE("Mode matching survives eigenvalue reordering and sign flips", "[modefollowing][tracking]") {
    Workspace work;
    Eigen::MatrixXd modes(3, 3); modes << 0,0,-1, 1,0,0, 0,1,0;
    Eigen::VectorXd previous(3); previous << 1,0,0;
    CHECK(mf::FindMode(modes, previous) == 2);
    CHECK(previous.isApprox(Eigen::Vector3d(1,0,0)));
}

TEST_CASE("Fixed-cell mode following reaches the analytic radial saddle", "[modefollowing][convergence]") {
    Workspace work;
    auto a = structure();
    GPIO io; io.readmap("index.map"); REQUIRE(io.write_one(a, "input.cfg", "mlpcfg"));
    std::ofstream("settings.ini") << "relax:driver fire\nrelax:car_mode true\nrelax:relax_cell false\n"
        "relax:maxiteration 1000\nrelax:gtol 1e-7\nrelax:fire:ts_init 0.001\nrelax:fire:ts_max 0.02\n";
    REQUIRE(work.run({"modefollowing", "settings.ini", "--potfile=pair.mtp", "--input=input.cfg",
        "--input-format=mlpcfg", "--nocell", "--car-rep", "--imode=0", "--mode=2", "--max-loop=5"}) == 0);
    Atoms result; REQUIRE(io.read_one(result, "POSCAR_SAD", "POSCAR"));
    double distance = (result.pos_[1]-result.pos_[0]).norm();
    Potentials::PotMTP pot("pair.mtp"); pot.getEFSOMP(result);
    CHECK(distance == Catch::Approx(7.0/3).margin(1e-7));
    CHECK(result.energy_ == Catch::Approx(256.0/27).margin(1e-8));
    CHECK(result.forces_[0].norm() < 1e-7);
    Eigen::VectorXd values; Eigen::MatrixXd vectors, h;
    Eigen::Matrix3d ref=result.cell_;
    hessian(result, pot, ref, values, vectors, h);
    CHECK(values(0) == Catch::Approx(-16).margin(1e-6));
    CHECK((values.array() < -1e-6).count() == 1);
    std::cout << "AUDIT SADDLE distance=" << std::setprecision(12) << distance
              << " energy=" << result.energy_ << " force=" << result.forces_[0].norm()
              << " eigenvalues=" << values.transpose() << '\n';
}

TEST_CASE("Small variable-cell Hessian does not index past the force array", "[modefollowing][small-cell]") {
    Workspace work;
    mf::nocell=false; mf::car_rep=false; mf::offset=6;
    const int count = GENERATE(1, 2, 3, 4, 5, 6, 7);
    auto a = structure();
    a.pos_.resize(count); a.chemmap_.assign(count, 0);
    for (int i=0; i<count; ++i) a.pos_[i] = Eigen::RowVector3d(2+1.3*i, 2+.1*i, 2+.2*i);
    Potentials::PotMTP pot("pair.mtp");
    Eigen::Matrix3d ref=a.cell_;
    Eigen::VectorXd values; Eigen::MatrixXd vectors, h;
    hessian(a, pot, ref, values, vectors, h);
    CHECK(h.allFinite());
}

TEST_CASE("Mode following escapes a minimum and reaches an analytic barrier", "[modefollowing][escape]") {
    Workspace work;
    {
        std::ifstream in("pair.mtp");
        std::string data((std::istreambuf_iterator<char>(in)), {});
        data.replace(data.find("radial_basis_size = 2"), 21, "radial_basis_size = 3");
        data.replace(data.find("{1, 1}"), 6, "{1.5, 2, 1}");
        std::ofstream("pair.mtp") << data;
    }
    // E(r)=(r-5)^2(r-2)^2: minimum r=2, maximum r=3.5, barrier=81/16.
    auto a=structure();
    a.pos_[1]=a.pos_[0]+Eigen::RowVector3d(2,0,0);
    GPIO io; io.readmap("index.map"); REQUIRE(io.write_one(a,"input.cfg","mlpcfg"));
    Potentials::PotMTP pot("pair.mtp"); pot.getEFSOMP(a);
    REQUIRE(a.energy_ == Catch::Approx(0).margin(1e-10));
    REQUIRE(a.forces_[0].norm() < 1e-10);
    auto barrier=a; barrier.pos_[1](0)+=1.5; pot.getEFSOMP(barrier);
    REQUIRE(barrier.energy_ == Catch::Approx(81.0/16));
    mf::nocell=true; mf::car_rep=true; mf::offset=0;
    Eigen::Matrix3d ref=a.cell_;
    Eigen::VectorXd values; Eigen::MatrixXd vectors,h;
    hessian(a,pot,ref,values,vectors,h);
    REQUIRE(values(5) == Catch::Approx(36));
    const double trial = vectors(3,5)-vectors(0,5)>0 ? 0.05 : -0.05;
    std::ofstream("settings.ini") << "relax:driver fire\nrelax:car_mode true\nrelax:relax_cell false\n"
        "relax:maxiteration 400\nrelax:gtol 1e-7\nrelax:fire:ts_init 0.001\nrelax:fire:ts_max 0.02\n";
    const int status=work.run({"modefollowing","settings.ini","--potfile=pair.mtp","--input=input.cfg",
        "--input-format=mlpcfg","--nocell","--car-rep","--imode=5","--mode=2","--max-loop=40",
        "--trial-step="+std::to_string(trial)});
    Atoms result; REQUIRE(io.read_one(result,"POSCAR_SAD","POSCAR"));
    pot.getEFSOMP(result);
    const double r=(result.pos_[1]-result.pos_[0]).norm();
    hessian(result,pot,ref,values,vectors,h);
    std::cout << "AUDIT ESCAPE status=" << status << " r=" << std::setprecision(12) << r
              << " E=" << result.energy_ << " force=" << result.forces_[0].norm()
              << " eigenvalues=" << values.transpose() << '\n';
    CHECK(status == 0);
    CHECK(r == Catch::Approx(3.5).margin(1e-6));
    CHECK(result.energy_ == Catch::Approx(81.0/16).margin(1e-8));
    CHECK(result.forces_[0].norm() < 1e-7);
    CHECK(values(0) == Catch::Approx(-18).margin(1e-6));
    CHECK((values.array() < -1e-6).count() == 1);
}

TEST_CASE("Fixed-cell single Newton step has the analytic displacement", "[modefollowing][newton]") {
    Workspace work;
    auto a=structure();
    GPIO io; io.readmap("index.map"); REQUIRE(io.write_one(a,"input.cfg","mlpcfg"));
    std::ofstream("settings.ini") << "relax:driver fire\nrelax:car_mode true\nrelax:relax_cell false\n";
    REQUIRE(work.run({"modefollowing","settings.ini","--potfile=pair.mtp","--input=input.cfg",
        "--input-format=mlpcfg","--nocell","--car-rep","--mode=4"}) == 0);
    Atoms result; REQUIRE(io.read_one(result,"POSCAR_NR_STEP","POSCAR"));
    const double r=(a.pos_[1]-a.pos_[0]).norm();
    const double expected=r-(r-5)*(3*r-7)/(6*r-22);
    CHECK((result.pos_[1]-result.pos_[0]).norm() == Catch::Approx(expected).margin(1e-8));
}

TEST_CASE("Symmetric cell yy mode has the same physical tangent in the optimizer", "[modefollowing][projection]") {
    Workspace work;
    auto a = structure(3);
    mf::nocell=false; mf::car_rep=false; mf::offset=6;
    Eigen::MatrixXd modes = Eigen::MatrixXd::Identity(24,24);
    Eigen::Matrix3d ref=a.cell_, invref=ref.inverse();
    auto nudged=a;
    constexpr double delta=1e-6;
    mf::NugeMode(nudged, modes, ref, invref, 1, delta);
    Potentials::PotMTP pot("pair.mtp");
    GPSettings settings;
    settings.Modify("relax_cell", "true");
    settings.Modify("driver", "fire");
    RelaxationDrvV2 opt(settings);
    mf::init_relaxation(opt, pot, a, ref, 0.0);
    REQUIRE(opt.x.size()==24);
    opt.x += delta * mf::optimizer_modes(modes).col(1);
    opt.X2LatPos();
    Eigen::Matrix3d expected=(nudged.cell_-a.cell_)/delta;
    Eigen::Matrix3d actual=(opt.atom.cell_-a.cell_)/delta;
    std::cout << "AUDIT YY TANGENT expected\n" << expected << "\noptimizer\n" << actual << '\n';
    CHECK(actual.isApprox(expected, 1e-4));
    for (int i=0; i<a.size(); ++i)
        CHECK(opt.atom.pos_[i].isApprox(nudged.pos_[i], 1e-10));
}

TEST_CASE("Mode-following derivatives include periodic images pressure and reference strain", "[modefollowing][derivatives]") {
    Workspace work;
    mf::nocell = GENERATE(true, false);
    mf::car_rep = GENERATE(true, false);
    mf::offset = mf::nocell ? 0 : (mf::car_rep ? 9 : 6);
    const double pressure = GENERATE(0.0, 0.02);
    auto a = structure();
    a.cell_ << 5.7, 0.3, 0.2, 0.1, 6.4, 0.4, 0.2, 0.1, 7.3;
    a.pos_ = {{0.8, 1.1, 1.2}, {3.1, 1.3, 1.5}, {1.4, 3.7, 2.9}};
    a.chemmap_ = {0, 0, 0};
    Eigen::Matrix3d ref=a.cell_, invref=ref.inverse();
    Eigen::Matrix3d deformation;
    deformation << 1.03, .01, -.02, .01, .98, .015, -.02, .015, 1.02;
    Eigen::Matrix3d deformed = ref*deformation;
    a.set_cell(deformed);
    Potentials::PotMTP pot("pair.mtp");
    Eigen::VectorXd values; Eigen::MatrixXd vectors, h;
    mf::compute_hessian(a, ref, pressure, pot, 1e-7, values, vectors, h);
    pot.getEFSOMP(a);
    const auto g=mf::gradient(a,ref,pressure);
    Eigen::RowVectorXd x;
    if (mf::car_rep) mf::LatPos2X(a,x); else mf::LatPos2x(a,invref,x);
    const auto evaluate = [&](Eigen::RowVectorXd q) {
        auto trial=a;
        if (mf::car_rep) mf::X2LatPos(trial,q); else mf::x2LatPos(trial,ref,q);
        pot.getEFSOMP(trial);
        return trial;
    };
    // Explicit periodic pair sum is independent of the MTP neighborhood/Hessian code.
    double pair_energy=0;
    for (int i=0;i<a.size();++i) for (int j=0;j<a.size();++j)
        for (int nx=-1;nx<=1;++nx) for (int ny=-1;ny<=1;++ny) for (int nz=-1;nz<=1;++nz) {
            if (i==j && nx==0 && ny==0 && nz==0) continue;
            const double r=(a.pos_[j]-a.pos_[i]+Eigen::RowVector3d(nx,ny,nz)*a.cell_).norm();
            if (r<5) pair_energy += .5*(r-5)*(r-5)*(r-1);
        }
    REQUIRE(a.energy_ == Catch::Approx(pair_energy).margin(1e-10));
    constexpr double delta=1e-6;
    for (int j=0;j<x.size();++j) {
        auto xp=x,xm=x; xp(j)+=delta; xm(j)-=delta;
        const auto plus=evaluate(xp),minus=evaluate(xm);
        const double fd=(plus.energy_-minus.energy_+pressure*(plus.get_volume()-minus.get_volume()))/(2*delta);
        INFO("Cartesian=" << mf::car_rep << " fixed=" << mf::nocell << " pressure=" << pressure << " column=" << j);
        CHECK(g(j) == Catch::Approx(fd).margin(2e-6));
        Eigen::VectorXd hfd=(mf::gradient(plus,ref,pressure)-mf::gradient(minus,ref,pressure))/(2*delta);
        CHECK((h.col(j)-hfd).cwiseAbs().maxCoeff()<2e-4);
    }
    GPSettings settings; settings.Modify("driver","fire");
    RelaxationDrvV2 opt(settings);
    mf::init_relaxation(opt,pot,a,ref,pressure);
    opt.eval();
    const Eigen::VectorXd initial=opt.x, og=opt.g;
    CHECK(opt.f == Catch::Approx(a.energy_+pressure*a.get_volume()));
    // Differentiate the optimizer's actual coordinates, not the mapping helper.
    for (int j=0;j<initial.size();++j) {
        opt.x=initial; opt.x(j)+=delta; opt.eval(); const double ep=opt.f;
        opt.x=initial; opt.x(j)-=delta; opt.eval(); const double em=opt.f;
        INFO("optimizer coordinate " << j);
        CHECK(og(j) == Catch::Approx((ep-em)/(2*delta)).margin(2e-6));
    }
}

TEST_CASE("Variable-cell mode following converges with either coordinate representation", "[modefollowing][vc-convergence]") {
    Workspace work;
    const bool cartesian=GENERATE(true,false);
    const bool rebase=GENERATE(false,true);
    auto a=structure();
    a.pos_[1]=a.pos_[0]+Eigen::RowVector3d(7.0/3-0.001,0,0);
    GPIO io; io.readmap("index.map"); REQUIRE(io.write_one(a,"input.cfg","mlpcfg"));
    std::ofstream settings("settings.ini");
    settings << "relax:driver fire\nrelax:maxiteration 2000\nrelax:gtol 1e-6\n"
        << "relax:fire:ts_init " << (cartesian ? "0.001" : "0.00001")
        << "\nrelax:fire:ts_max " << (cartesian ? "0.02" : "0.0005") << '\n';
    settings.close();
    std::vector<std::string> args{"modefollowing","settings.ini","--potfile=pair.mtp","--input=input.cfg",
        "--input-format=mlpcfg","--imode=0","--mode=2","--max-loop=5","--eigzero-tol=0.001"};
    if (cartesian) args.push_back("--car-rep");
    if (rebase) args.push_back("--updateref");
    REQUIRE(work.run(args)==0);
    Atoms result; REQUIRE(io.read_one(result,"POSCAR_SAD","POSCAR"));
    Potentials::PotMTP pot("pair.mtp"); pot.getEFSOMP(result);
    CHECK((result.pos_[1]-result.pos_[0]).norm()==Catch::Approx(7.0/3).margin(2e-7));
    CHECK(result.energy_==Catch::Approx(256.0/27).margin(1e-9));
    CHECK(result.forces_[0].norm()<1e-6);
}

namespace {
struct CommandFixture {
    Workspace work;
    GPIO io;
    Atoms atom=structure();
    Eigen::Matrix3d ref=atom.cell_, invref=ref.inverse();
    Eigen::RowVectorXd initial;
    std::vector<std::string> args;
    CommandFixture(bool fixed, bool cartesian) {
        mf::nocell=fixed; mf::car_rep=cartesian;
        mf::offset=fixed ? 0 : (cartesian ? 9 : 6);
        io.readmap("index.map");
        if (!io.write_one(atom,"input.cfg","mlpcfg")) throw std::runtime_error("write input failed");
        if (cartesian) mf::LatPos2X(atom,initial); else mf::LatPos2x(atom,invref,initial);
        Eigen::MatrixXd h=Eigen::VectorXd::LinSpaced(initial.size(),1,initial.size()).asDiagonal();
        std::ofstream("hessian") << std::setprecision(17) << h << '\n';
        std::ofstream("settings.ini") << "relax:driver fire\n";
        args={"modefollowing","settings.ini","--potfile=pair.mtp","--input=input.cfg",
              "--input-format=mlpcfg","--hessian=hessian"};
        if (fixed) args.push_back("--nocell");
        if (cartesian) args.push_back("--car-rep");
    }
    Eigen::RowVectorXd coordinates(Atoms a) {
        Eigen::RowVectorXd x;
        if (mf::car_rep) mf::LatPos2X(a,x); else mf::LatPos2x(a,invref,x);
        return x;
    }
    Atoms decode(Eigen::RowVectorXd x) {
        auto a=atom;
        if (mf::car_rep) mf::X2LatPos(a,x); else mf::x2LatPos(a,ref,x);
        return a;
    }
    int run(std::initializer_list<std::string> extra) {
        auto command=args; command.insert(command.end(),extra.begin(),extra.end());
        return work.run(command);
    }
};
struct Capture {
    std::ostringstream stream;
    std::streambuf *previous=std::cout.rdbuf(stream.rdbuf());
    ~Capture() { std::cout.rdbuf(previous); }
    std::map<int,double> pairs() {
        std::map<int,double> values;
        std::istringstream lines(stream.str()); std::string line;
        while (std::getline(lines,line)) {
            std::istringstream row(line); int index; double value; std::string rest;
            if (row>>index>>value && !(row>>rest)) values[index]=value;
        }
        return values;
    }
};
}

TEST_CASE("Modes 1 and 6 reproduce physical displacements in all coordinate conventions", "[modefollowing][animation][nudge]") {
    const bool fixed=GENERATE(true,false), cartesian=GENERATE(true,false);
    CommandFixture f(fixed,cartesian);
    const int selected=fixed ? 3 : 1;
    REQUIRE(f.run({"--mode=6","--imode="+std::to_string(selected),"--trial-step=0.001"})==0);
    Atoms nudged; REQUIRE(f.io.read_one(nudged,"POSCAR_NUGED","POSCAR"));
    Eigen::RowVectorXd expected=f.initial; expected(selected)+=.001;
    CHECK(f.coordinates(nudged).isApprox(expected,1e-12));
    const std::string trajectory=GENERATE("linear","half","cycle");
    auto args=f.args;
    args.insert(args.end(),{"--mode=1","--amplitude=0.001","--frames=5","--request-efs=true"});
    if (trajectory=="linear") args.push_back("--linetraj");
    if (trajectory=="cycle") args.push_back("--cycle");
    REQUIRE(f.work.run(args)==0);
    std::vector<Atoms> frames;
    REQUIRE(f.io.read(frames,"vibmode_meta_"+std::to_string(selected)+".xyz","extxyz"));
    REQUIRE(frames.size()==5);
    for (int i=0;i<5;++i) {
        double phi=trajectory=="linear" ? -1.0+0.5*i :
            std::sin(-M_PI/2+(trajectory=="cycle" ? 2*M_PI*i/5 : M_PI*i/4));
        expected=f.initial; expected(selected)+=.001*phi;
        INFO("frame=" << i << " trajectory=" << trajectory << " fixed=" << fixed << " cartesian=" << cartesian);
        CHECK(f.coordinates(frames[i]).isApprox(expected,1e-8));
        CHECK(frames[i].energy_==Catch::Approx(analytic_energy(frames[i])).margin(1e-7));
    }
}

TEST_CASE("Mode 3 projects matching structures in the Hessian coordinates", "[modefollowing][displacement-projection]") {
    CommandFixture f(GENERATE(true,false),GENERATE(true,false));
    auto target=f.initial;
    target.tail(6) += Eigen::RowVectorXd::LinSpaced(6,.0001,.0006);
    if (!mf::nocell) target.head(mf::offset) += Eigen::RowVectorXd::LinSpaced(mf::offset,.0001,.0006);
    auto atom2=f.decode(target);
    REQUIRE(f.io.write_one(atom2,"second.cfg","mlpcfg"));
    Capture output;
    REQUIRE(f.run({"--mode=3","--input2=second.cfg"})==0);
    auto projection=output.pairs(); REQUIRE(projection.size()==target.size());
    for (int i=0;i<target.size();++i)
        CHECK(projection.at(i)==Catch::Approx(f.initial(i)-target(i)).margin(1e-9));
}

TEST_CASE("Modes 4 and 8 use the unfiltered enthalpy gradient", "[modefollowing][newton][gradient-projection]") {
    CommandFixture f(GENERATE(true,false),GENERATE(true,false));
    constexpr double pressure=.013, delta=1e-6;
    const int n=f.initial.size();
    Eigen::MatrixXd h=(1e6*Eigen::VectorXd::LinSpaced(n,1,n)).asDiagonal();
    std::ofstream("hessian") << std::setprecision(17) << h << '\n';
    Eigen::RowVectorXd finite(n);
    for (int i=0;i<n;++i) {
        auto xp=f.initial,xm=f.initial; xp(i)+=delta; xm(i)-=delta;
        auto plus=f.decode(xp),minus=f.decode(xm);
        finite(i)=(analytic_energy(plus)-analytic_energy(minus)+pressure*(plus.get_volume()-minus.get_volume()))/(2*delta);
    }
    const std::string pressure_arg="--pressure="+fmt::format("{:.17g}",pressure*eVA3_to_GPa);
    {
        Capture output;
        REQUIRE(f.run({"--mode=8",pressure_arg})==0);
        auto projection=output.pairs(); REQUIRE(projection.size()==n);
        for (int i=0;i<n;++i)
            CHECK(projection.at(i)==Catch::Approx(finite(i)).margin(2e-6));
    }
    REQUIRE(f.run({"--mode=4",pressure_arg})==0);
    Atoms result; REQUIRE(f.io.read_one(result,"POSCAR_NR_STEP","POSCAR"));
    Eigen::RowVectorXd expected=f.initial-finite.cwiseQuotient(h.diagonal().transpose());
    CHECK(f.coordinates(result).isApprox(expected,1e-10));
}

TEST_CASE("Mode 7 correlates identical eigensystems without changing coordinate bases", "[modefollowing][correlation]") {
    CommandFixture f(GENERATE(true,false),GENERATE(true,false));
    REQUIRE(f.io.write_one(f.atom,"second.cfg","mlpcfg"));
    auto args=f.args;
    args.erase(std::find(args.begin(),args.end(),"--hessian=hessian"));
    args.insert(args.end(),{"--mode=7","--input2=second.cfg"});
    Capture output;
    REQUIRE(f.work.run(args)==0);
    auto matches=output.pairs(); REQUIRE(matches.size()==f.initial.size());
    for (int i=0;i<f.initial.size();++i) CHECK(matches.at(i)==i);
}

TEST_CASE("Mode following rejects unsupported modes invalid parameters and malformed Hessians", "[modefollowing][validation]") {
    CommandFixture f(true,true);
    SECTION("unsupported or unimplemented mode") {
        const std::string mode=GENERATE("0","5","10","-1");
        REQUIRE_THROWS_WITH(f.run({"--mode="+mode}),Catch::Matchers::ContainsSubstring("Unsupported mode"));
    }
    SECTION("mode index") {
        const std::string mode=GENERATE("2","6","9");
        const std::string index=GENERATE("-1","6");
        REQUIRE_THROWS_WITH(f.run({"--mode="+mode,"--imode="+index}),Catch::Matchers::ContainsSubstring("out of range"));
    }
    SECTION("invalid numerical parameters") {
        const std::string argument=GENERATE("--max-loop=-1","--frames=1","--frames=-1","--eigzero-tol=0",
            "--eigzero-tol=nan","--symprec=0","--pressure=inf","--trial-step=nan","--amplitude=nan");
        REQUIRE_THROWS(f.run({"--mode=1",argument}));
    }
    SECTION("malformed hessian") {
        const std::string kind=GENERATE("short","extra","asymmetric","nan","missing");
        Eigen::MatrixXd h=Eigen::MatrixXd::Identity(6,6);
        if (kind=="asymmetric") h(0,1)=1;
        if (kind=="nan") h(0,0)=std::numeric_limits<double>::quiet_NaN();
        std::ofstream file("hessian");
        if (kind=="short") file<<"1 2 3\n";
        else file<<h<<'\n';
        if (kind=="extra") file<<"7\n";
        file.close();
        if (kind=="missing") std::filesystem::remove("hessian");
        REQUIRE_THROWS(f.run({"--mode=9","--imode=0"}));
    }
    SECTION("second input mismatch") {
        const std::string mode=GENERATE("3","7");
        const std::string kind=GENERATE("empty","count","species","cell");
        auto second=f.atom;
        if (kind=="count") { second.pos_.resize(1); second.chemmap_.resize(1); }
        if (kind=="species") second.chemmap_[0]=1;
        if (kind=="cell") second.cell_(0,0)+=.1;
        if (kind=="empty") std::ofstream("second.cfg")<<"\n";
        else REQUIRE(f.io.write_one(second,"second.cfg","mlpcfg"));
        REQUIRE_THROWS(f.run({"--mode="+mode,"--input2=second.cfg"}));
    }
    SECTION("incompatible optimizer") {
        const std::string driver=GENERATE("lbfgs", "lbfgspp", "cg", "gd");
        std::ofstream("settings.ini")<<"relax:driver "<<driver<<'\n';
        REQUIRE_THROWS_WITH(f.run({"--mode=2","--imode=0"}),Catch::Matchers::ContainsSubstring("requires relax:driver fire"));
    }
}

TEST_CASE("Mode following propagates all file output failures", "[modefollowing][output]") {
    CommandFixture f(true,true);
    const std::string mode=GENERATE("1","4","6","9");
    std::string filename;
    if (mode=="1") filename=GENERATE("EigensystemFull.dat","EIGSYS","vibmode_meta_0.xyz");
    else filename=mode=="4" ? "POSCAR_NR_STEP" : mode=="6" ? "POSCAR_NUGED" : "MODECAR";
    std::filesystem::create_symlink("/dev/full",filename);
    REQUIRE_THROWS(f.run({"--mode="+mode,"--imode=0","--frames=2"}));
}

TEST_CASE("Mode 1 writes a complete reusable eigensystem and mode 9 accepts its Hessian", "[modefollowing][eigensystem]") {
    CommandFixture f(GENERATE(true,false),GENERATE(true,false));
    auto args=f.args;
    args.erase(std::find(args.begin(),args.end(),"--hessian=hessian"));
    args.insert(args.end(),{"--mode=1","--frames=2","--amplitude=0"});
    REQUIRE(f.work.run(args)==0);
    const int n=f.initial.size();
    Eigen::MatrixXd h(n,n),q(n,n),q2(n,n); Eigen::VectorXd values(n);
    std::ifstream hf("hessianfull"),vf("EigensystemFull.dat"),qf("EIGSYS");
    for (int i=0;i<n;++i) {
        vf>>values(i);
        for (int j=0;j<n;++j) { hf>>h(i,j); vf>>q(j,i); qf>>q2(j,i); }
    }
    REQUIRE(hf.good()); REQUIRE(vf.good()); REQUIRE(qf.good());
    CHECK(q.isApprox(q2,1e-14));
    CHECK((q.transpose()*q-Eigen::MatrixXd::Identity(n,n)).norm()<1e-12);
    CHECK((h*q-q*values.asDiagonal()).norm()<1e-10*std::max(1.0,h.norm()));
    REQUIRE(f.run({"--hessian=hessianfull","--mode=9","--imode=0"})==0);
    CHECK(std::filesystem::file_size("MODECAR")>0);
}

TEST_CASE("Mode 7 compares nondegenerate modes in one reference cell", "[modefollowing][correlation]") {
    CommandFixture f(false,false);
    f.atom.pos_[1]=f.atom.pos_[0]+Eigen::RowVector3d(7.0/3,0,0);
    REQUIRE(f.io.write_one(f.atom,"input.cfg","mlpcfg"));
    auto second=f.atom;
    Eigen::Matrix3d strain;
    strain << 1.04,0,0, .02,.97,0, -.01,.01,1.02;
    second.cell_=f.ref*strain; // Keep the stationary Cartesian pair unchanged.
    auto radial_tangent=[&](Atoms atom) {
        Eigen::RowVectorXd x; mf::LatPos2x(atom,f.invref,x);
        Eigen::VectorXd tangent(x.size());
        for (int j=0;j<x.size();++j) {
            auto xp=x,xm=x; xp(j)+=1e-6; xm(j)-=1e-6;
            auto ap=atom,am=atom;
            mf::x2LatPos(ap,f.ref,xp); mf::x2LatPos(am,f.ref,xm);
            tangent(j)=((ap.pos_[1]-ap.pos_[0]).norm()-(am.pos_[1]-am.pos_[0]).norm())/2e-6;
        }
        return tangent.normalized().eval();
    };
    const double expected=std::abs(radial_tangent(f.atom).dot(radial_tangent(second)));
    if (GENERATE(false, true)) {
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.4, Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
        second.cell_ *= rotation;
        for (auto &position : second.pos_) position *= rotation;
    }
    REQUIRE(f.io.write_one(second,"second.cfg","mlpcfg"));
    auto args=f.args;
    args.erase(std::find(args.begin(),args.end(),"--hessian=hessian"));
    args.insert(args.end(),{"--mode=7","--input2=second.cfg"});
    Capture output; REQUIRE(f.work.run(args)==0);
    std::istringstream lines(output.stream.str()); std::string line;
    bool found=false;
    while(std::getline(lines,line)) {
        std::istringstream row(line); std::vector<double> numbers;
        double value; while(row>>value) numbers.push_back(value);
        if (numbers.size()==f.initial.size()+1 && numbers[0]==0) {
            CHECK(numbers[1]==Catch::Approx(expected).margin(6e-7));
            found=true; break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("Metric mode following matches energy derivatives away from the reference cell", "[modefollowing][hessian][metric]") {
    Workspace work;
    mf::nocell = false; mf::car_rep = false; mf::offset = 6;
    auto atom = structure();
    Eigen::Matrix3d ref = atom.cell_, inverse = ref.inverse();
    Eigen::RowVectorXd x;
    mf::LatPos2x(atom, inverse, x);
    x.head<6>() << .08, -.03, .05, .02, -.015, .025;
    mf::x2LatPos(atom, ref, x);
    Potentials::PotMTP pot("pair.mtp");
    Eigen::VectorXd values; Eigen::MatrixXd vectors, hessian;
    constexpr double pressure = .0003, delta = 1e-5;
    mf::compute_hessian(atom, ref, pressure, pot, 1e-7, values, vectors, hessian);
    const auto gradient = mf::gradient(atom, ref, pressure);
    const auto energy = [&](Eigen::RowVectorXd q) {
        auto trial = atom;
        mf::x2LatPos(trial, ref, q);
        return analytic_energy(trial) + pressure * trial.get_volume();
    };
    for (int i = 0; i < x.size(); ++i) {
        auto plus = x, minus = x;
        plus(i) += delta; minus(i) -= delta;
        REQUIRE(gradient(i) == Catch::Approx((energy(plus)-energy(minus))/(2*delta)).margin(1e-6));
        for (int j = 0; j < x.size(); ++j) {
            auto pp=x, pm=x, mp=x, mm=x;
            pp(i)+=delta; pp(j)+=delta; pm(i)+=delta; pm(j)-=delta;
            mp(i)-=delta; mp(j)+=delta; mm(i)-=delta; mm(j)-=delta;
            const double numerical = (energy(pp)-energy(pm)-energy(mp)+energy(mm))/(4*delta*delta);
            REQUIRE(hessian(i,j) == Catch::Approx(numerical).margin(.003));
        }
    }
}
