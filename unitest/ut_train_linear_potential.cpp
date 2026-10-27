// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/train_linear_potential.h"
#include "potentials/potential_descriptor_linear.h"
#include <cmath>
#include <limits>
#include <stdexcept>

using Catch::Approx;

#ifdef IMRMLP_MPI
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 20261002;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    const int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}
#endif

namespace {
class ConstantPotential : public Potentials::AnyPotential {
public:
    double coefficient = 7;
    double descriptor = 1;
    int saves = 0;
    bool fail_save = false;
    ConstantPotential() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    bool supports_linear_fit() const override { return true; }
    int get_NL_coeffs_count() const override { return 1; }
    void nl_set_coeffs(const double *x) override { coefficient = x[0]; }
    void linear_compute_descriptor(Neighborhood &, double *out) const override { out[0] = descriptor; }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        atom.resetEFS();
        atom.energy_ = coefficient * atom.size();
    }
    void Save(const std::string &) override {
        ++saves;
        if (fail_save) throw std::runtime_error("test model save failed");
    }
};

struct Data {
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> nbhs;
    Data() {
        for (double energy : {0., 8.}) {
            Atoms atom;
            atom.cell_ = Eigen::Matrix3d::Identity() * 20;
            atom.pos_ = {{1, 1, 1}, {3, 1, 1}};
            atom.chemmap_ = {0, 0};
            atom.resetEFS();
            atom.energy_ = energy;
            atom.forces_.assign(2, Eigen::Vector3d(3, 4, 0));
            atom.stress_.diagonal() << 1, 2, 2;
            atoms.push_back(atom);
            nbhs.emplace_back(atoms.back(), 3);
        }
    }
};

GPSettings settings()
{
    GPSettings s;
    s.Modify("force_weight", "0");
    s.Modify("stress_weight", "0");
    s.Modify("weight_L2", "0");
    return s;
}
}

TEST_CASE("Linear training rejects invalid state and data without changing coefficients", "[train][linear]")
{
    ConstantPotential pot;
    TrainerLinearPotential trainer(settings());
    Data data;
    REQUIRE(trainer.emonitor.erms == 0);
    REQUIRE_THROWS(trainer.Assemble());
    REQUIRE_THROWS(trainer.Solve());
    REQUIRE_THROWS(trainer.Train());
    REQUIRE_THROWS(trainer.save_current("unused"));
    REQUIRE_THROWS(trainer.init(data.atoms, data.nbhs, nullptr));
    std::vector<Atoms> empty;
    std::vector<Neighborhoods> no_nbhs;
    REQUIRE_THROWS(trainer.init(empty, no_nbhs, &pot));
    REQUIRE_THROWS(trainer.init(data.atoms, no_nbhs, &pot));
    REQUIRE(pot.coefficient == 7);

    trainer.init(data.atoms, data.nbhs, &pot);
    REQUIRE_THROWS(trainer.Solve());
    trainer.Assemble();
    trainer.Solve();
    REQUIRE(pot.coefficient == Approx(2));
    trainer.init(data.atoms, data.nbhs, &pot);
    REQUIRE_THROWS(trainer.Solve());
    data.atoms[0].forces_.resize(1);
    REQUIRE_THROWS(trainer.init(data.atoms, data.nbhs, &pot));
    REQUIRE_THROWS(trainer.Assemble());
    REQUIRE(pot.coefficient == Approx(2));
}

TEST_CASE("Linear training reports errors of the fitted potential in both workflows", "[train][linear]")
{
    const bool one_step = GENERATE(false, true);
    ConstantPotential pot;
    TrainerLinearPotential trainer(settings());
    Data data;
    trainer.init(data.atoms, data.nbhs, &pot);
    trainer.init_valid(data.atoms, data.nbhs);
    trainer.monit_valid_err = true;
    if (one_step) trainer.Train();
    else { trainer.Assemble(); trainer.Solve(); }
    REQUIRE(pot.coefficient == Approx(2));
    REQUIRE(trainer.emonitor.erms == Approx(2));
    REQUIRE(trainer.emonitor.frms == Approx(5));
    REQUIRE(trainer.emonitor.srms == Approx(3));
    REQUIRE(pot.saves == (one_step && mpi.rank == 0 ? 1 : 0));
    if (one_step) {
        REQUIRE(trainer.emonitor_valid.erms == Approx(2));
        REQUIRE(trainer.emonitor_valid.frms == Approx(5));
        REQUIRE(trainer.emonitor_valid.srms == Approx(3));
    }
    for (auto &atom : data.atoms) atom.energy_ = 4;
    trainer.init(data.atoms, data.nbhs, &pot);
    trainer.Assemble();
    trainer.Solve();
    REQUIRE(trainer.emonitor.erms == Approx(0).margin(1e-12));
    REQUIRE(trainer.emonitor.frms == Approx(5));
}

TEST_CASE("Linear training propagates root model save failure to every rank", "[train][linear][save]")
{
    ConstantPotential pot;
    pot.fail_save = mpi.rank == 0;
    TrainerLinearPotential trainer(settings());
    Data data;
    trainer.init(data.atoms, data.nbhs, &pot);
    std::string expected = "test model save failed";
#ifdef IMRMLP_MPI
    expected = "Training failure on rank 0: " + expected;
#endif
    REQUIRE_THROWS_WITH(trainer.Train(), expected);
    REQUIRE(pot.coefficient == Approx(2));
    REQUIRE(pot.saves == (mpi.rank == 0 ? 1 : 0));
}

TEST_CASE("Linear training rejects nonfinite labels and invalid fitness", "[train][linear]")
{
    ConstantPotential pot;
    TrainerLinearPotential trainer(settings());
    Data data;
    const int field = GENERATE(0, 1, 2, 3, 4, 5);
    // Only one rank has bad data: every rank must receive the failure.
    if (mpi.rank == 0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (field) {
        case 0: data.atoms[0].energy_ = nan; break;
        case 1: data.atoms[0].forces_[0](0) = nan; break;
        case 2: data.atoms[0].stress_(1, 1) = nan; break;
        case 3: data.atoms[0].fitness = nan; break;
        case 4: data.atoms[0].atomic_fitness = {1, nan}; break;
        case 5: data.atoms[0].atomic_fitness = {1, -1}; break;
        }
    }
    REQUIRE_THROWS(trainer.init(data.atoms, data.nbhs, &pot));
    REQUIRE(pot.coefficient == 7);
    REQUIRE(pot.saves == 0);
}

TEST_CASE("Linear training refuses absent or disabled labels", "[train][linear]")
{
    ConstantPotential pot;
    auto params = settings();
    Data data;
    const int mode = GENERATE(0, 1, 2);
    if (mode == 0)
        for (auto &atom : data.atoms) atom.clearEFS();
    else if (mode == 1) params.Modify("energy_weight", "0");
    else
        for (auto &atom : data.atoms) atom.fitness = 0;
    TrainerLinearPotential trainer(params);
    trainer.init(data.atoms, data.nbhs, &pot);
    REQUIRE_THROWS_WITH(trainer.Train(), Catch::Matchers::ContainsSubstring("no effective labels"));
    REQUIRE(pot.coefficient == 7);
    REQUIRE(pot.saves == 0);
}

TEST_CASE("Linear training rejects invalid weights and nonfinite assembly", "[train][linear]")
{
    ConstantPotential pot;
    auto params = settings();
    const int mode = GENERATE(0, 1, 2, 3);
    if (mode == 0) params.Modify("energy_weight", "-1");
    if (mode == 1) params.Modify("weight_L2", "-1");
    if (mode == 2 && mpi.rank == 0) pot.descriptor = std::numeric_limits<double>::infinity();
    if (mode == 3) params.Modify("energy_weight", "1e308");
    TrainerLinearPotential trainer(params);
    Data data;
    trainer.init(data.atoms, data.nbhs, &pot);
    REQUIRE_THROWS(trainer.Train());
    REQUIRE(pot.coefficient == 7);
    REQUIRE(pot.saves == 0);
}

TEST_CASE("Linear training rejects a nonfinite solution before replacing coefficients", "[train][linear]")
{
    ConstantPotential pot;
    pot.descriptor = 1e-150;
    Data data;
    for (auto &atom : data.atoms) atom.energy_ = 1e160;
    TrainerLinearPotential trainer(settings());
    trainer.init(data.atoms, data.nbhs, &pot);
    trainer.Assemble(); // Both the normal matrix and target remain finite.
    REQUIRE_THROWS_WITH(trainer.Solve(), Catch::Matchers::ContainsSubstring("nonfinite coefficients"));
    REQUIRE(pot.coefficient == 7);
}

TEST_CASE("Linear stress fitting uses a rotation invariant tensor norm", "[train][linear][stress]")
{
    GPSettings descriptor_settings;
    descriptor_settings.Modify("descriptor:dtype", "behler");
    descriptor_settings.Modify("descriptor:behler:dispcount", "1");
    descriptor_settings.Modify("descriptor:behler:dispselect", "1");
    descriptor_settings.Modify("descriptor:behler:cutoff", "5");
    auto params = settings();
    params.Modify("energy_weight", "0");
    params.Modify("stress_weight", "1");

    double unrotated_coefficient = 0;
    for (const bool rotated : {false, true}) {
        Potentials::PotentialDescriptorLinear pot(descriptor_settings);
        double coefficient = 1;
        pot.nl_set_coeffs(&coefficient);
        Atoms atom;
        atom.cell_ = 10 * Eigen::Matrix3d::Identity();
        atom.pbc_ = {false, false, false};
        atom.pos_ = {{3, 3, 3}, rotated
            ? Eigen::RowVector3d(3 + std::sqrt(2.), 3 + std::sqrt(2.), 3)
            : Eigen::RowVector3d(5, 3, 3)};
        atom.chemmap_ = {0, 0};
        Neighborhoods nbh(atom, pot.cutoff);
        pot.getEFS(atom, nbh);
        const Eigen::Matrix3d basis_stress = atom.stress_;
        // An isotropic target is unchanged by rotating the configuration.
        atom.stress_ = Eigen::Matrix3d::Identity();
        const double expected = basis_stress.trace() / basis_stress.squaredNorm();
        std::vector<Atoms> atoms{atom};
        std::vector<Neighborhoods> nbhs{nbh};
        TrainerLinearPotential trainer(params);
        trainer.init(atoms, nbhs, &pot);
        trainer.Assemble();
        trainer.Solve();
        pot.nl_get_coeffs(&coefficient);
        REQUIRE(coefficient == Approx(expected).epsilon(1e-10));
        if (!rotated) unrotated_coefficient = coefficient;
        else REQUIRE(coefficient == Approx(unrotated_coefficient).epsilon(1e-10));
    }
}

#ifdef IMRMLP_MPI
TEST_CASE("Linear training permits empty local MPI shards", "[train][linear][mpi]")
{
    ConstantPotential pot;
    TrainerLinearPotential trainer(settings());
    Data data;
    if (mpi.rank != 0) { data.atoms.clear(); data.nbhs.clear(); }
    trainer.init(data.atoms, data.nbhs, &pot);
    if (mpi.rank == 0) {
        trainer.init_valid(data.atoms, data.nbhs);
        trainer.monit_valid_err = true;
    }
    trainer.Train();
    REQUIRE(pot.coefficient == Approx(2));
    REQUIRE(trainer.emonitor.erms == Approx(2));
    REQUIRE(trainer.emonitor_valid.erms == Approx(2));
}
#endif

namespace {
class PairQuadraticPotential : public Potentials::AnyPotential {
public:
    Eigen::Vector2d coefficients{7., -3.};
    bool bad_gradient = false;
    PairQuadraticPotential() { cutoff=8.; cutoffsq=64.; species_count=1; set_inited(); }
    bool supports_linear_fit() const override { return true; }
    int get_NL_coeffs_count() const override { return 2; }
    void nl_get_coeffs(double *p) const override { Eigen::Map<Eigen::Vector2d> output(p); output=coefficients; }
    void nl_set_coeffs(const double *p) override { coefficients=Eigen::Map<const Eigen::Vector2d>(p); }
    void Save(const std::string &) override {}
    void linear_compute_descriptor(Neighborhood &n, double *d) const override {
        d[0]=d[1]=0.;
        for (int j=0;j<n.nbhcount_;++j) {
            d[0]+=.25*n.pos_[j].squaredNorm();
            d[1]+=.25*n.pos_[j][0]*n.pos_[j][0];
        }
    }
    void linear_compute_descriptor_grad(Neighborhood &n,double *d,double *g) const override {
        linear_compute_descriptor(n,d);
        for(int f=0;f<2;++f) for(int j=0;j<n.nbhcount_;++j) for(int a=0;a<3;++a)
            g[f*n.nbhcount_*3+j*3+a]=bad_gradient ? std::numeric_limits<double>::infinity()
                : .5*n.pos_[j][a]*(f==0 || a==0);
    }
    void getEFS(Atoms &a,AnyNeighborhoods &nbhs) override {
        a.resetEFS();
        for(int i=0;i<a.size();++i) {
            auto &n=nbhs[i]; double d[2]; linear_compute_descriptor(n,d);
            a.energy_+=coefficients.dot(Eigen::Map<Eigen::Vector2d>(d));
            for(int j=0;j<n.nbhcount_;++j) {
                Eigen::Vector3d f=.5*coefficients[0]*n.pos_[j].transpose();
                f[0]+=.5*coefficients[1]*n.pos_[j][0];
                a.forces_[i]+=f; a.forces_[n.indices_[j]]-=f;
                a.stress_-=f*n.pos_[j];
            }
        }
    }
};
Atoms pair_hvp_atom(double isotropic=1.7,double axial=-.4,int probes=2) {
    Atoms a; a.cell_=20.*Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    a.pos_={{2.,2.,2.},{3.8,2.5,2.2},{2.4,4.,3.1}}; a.chemmap_={0,0,0};
    a.hvp_directions_.resize(9,probes); a.hvp_targets_.resize(9,probes);
    for(int k=0;k<probes;++k) {
        Eigen::Vector3d sum=Eigen::Vector3d::Zero();
        for(int i=0;i<3;++i) for(int axis=0;axis<3;++axis) {
            const double v=std::sin(.37*(3*i+axis+1)*(k+1));
            a.hvp_directions_(3*i+axis,k)=v; sum[axis]+=v;
        }
        for(int i=0;i<3;++i) for(int axis=0;axis<3;++axis)
            a.hvp_targets_(3*i+axis,k)=(isotropic+(axis==0 ? axial : 0.))
                *(3*a.hvp_directions_(3*i+axis,k)-sum[axis]);
    }
    return a;
}
GPSettings pair_hvp_settings() {
    auto s=settings(); s.Modify("energy_weight","0");
    s.Modify("hvp_weight",".7"); s.Modify("hvp_fd_step",".0001"); return s;
}
void set_force_difference_targets(Potentials::AnyPotential &pot,Atoms &a) {
    for(int k=0;k<a.hvp_directions_.cols();++k) {
        double max_norm=0.;
        for(int i=0;i<a.size();++i)
            max_norm=std::max(max_norm,a.hvp_directions_.col(k).segment<3>(3*i).norm());
        const double epsilon=1e-4/max_norm;
        Atoms plus=a,minus=a;
        for(int i=0;i<a.size();++i) {
            const Eigen::RowVector3d delta=epsilon*a.hvp_directions_.col(k).segment<3>(3*i).transpose();
            plus.pos_[i]+=delta; minus.pos_[i]-=delta;
        }
        Neighborhoods pn(plus,pot.cutoff),mn(minus,pot.cutoff);
        pot.getEFS(plus,pn); pot.getEFS(minus,mn);
        for(int i=0;i<a.size();++i)
            a.hvp_targets_.col(k).segment<3>(3*i)=-(plus.forces_[i]-minus.forces_[i]).transpose()/(2*epsilon);
    }
}
}

TEST_CASE("Linear HVP-only fit matches an independent analytic pair Hessian", "[train][linear][hvp]") {
    const bool train=GENERATE(false,true);
    PairQuadraticPotential pot;
    std::vector<Atoms> atoms{pair_hvp_atom()};
    const auto saved=atoms[0];
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    TrainerLinearPotential trainer(pair_hvp_settings()); trainer.init(atoms,nbhs,&pot);
    if(train) trainer.Train(); else { trainer.Assemble(); trainer.Solve(); }
    REQUIRE(pot.coefficients[0]==Approx(1.7).epsilon(1e-9));
    REQUIRE(pot.coefficients[1]==Approx(-.4).epsilon(1e-9));
    REQUIRE(trainer.emonitor.hvp_rms<1e-9);
    REQUIRE(trainer.emonitor.loss_hvp<1e-18);
    REQUIRE(atoms[0].pos_==saved.pos_);
    REQUIRE(atoms[0].hvp_targets_==saved.hvp_targets_);
}

TEST_CASE("Linear HVP assembly errors propagate collectively before replacing coefficients", "[train][linear][hvp]") {
    const int mode=GENERATE(0,1,2,3,4,5);
    PairQuadraticPotential pot;
    std::vector<Atoms> atoms{pair_hvp_atom()};
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    if(mpi.rank==0) {
        if(mode==0) atoms[0].hvp_targets_.resize(9,1);
        if(mode==1) atoms[0].hvp_directions_.col(0).setZero();
        if(mode==2) atoms[0].hvp_targets_(0,0)=std::numeric_limits<double>::quiet_NaN();
        if(mode==3) atoms[0].cartesian=false;
        if(mode==4) pot.bad_gradient=true;
        if(mode==5) atoms[0].hvp_directions_.col(0).setConstant(1e-320);
    }
    TrainerLinearPotential trainer(pair_hvp_settings()); trainer.init(atoms,nbhs,&pot);
    REQUIRE_THROWS(trainer.Assemble());
    REQUIRE(pot.coefficients==Eigen::Vector2d(7.,-3.));
    REQUIRE_THROWS(trainer.Solve());
}

TEST_CASE("Linear HVP normalization and reported errors match an analytic weighted fit", "[train][linear][hvp]") {
    const int repeats=GENERATE(1,3);
    PairQuadraticPotential pot;
    std::vector<Atoms> atoms{pair_hvp_atom(1.7,-.4),pair_hvp_atom(.8,.3)};
    atoms[0].fitness=.4; atoms[1].fitness=1.8;
    // Different probe counts and repeated columns must not change a structure's weight.
    for(int s=0;s<2;++s) {
        const Eigen::MatrixXd directions=atoms[s].hvp_directions_;
        const Eigen::MatrixXd targets=atoms[s].hvp_targets_;
        const int count=repeats*(s+1);
        atoms[s].hvp_directions_.resize(9,2*count);
        atoms[s].hvp_targets_.resize(9,2*count);
        for(int k=0;k<count;++k) {
            atoms[s].hvp_directions_.middleCols(2*k,2)=directions;
            atoms[s].hvp_targets_.middleCols(2*k,2)=targets;
        }
        atoms[s].atomic_fitness.assign(3,0.); // HVP uses structure fitness only.
    }
    auto rows=[](const Atoms &a,int k) {
        Eigen::Matrix<double,9,2> design=Eigen::Matrix<double,9,2>::Zero();
        for(int axis=0;axis<3;++axis) {
            double sum=0.;
            for(int i=0;i<3;++i) sum+=a.hvp_directions_(3*i+axis,k);
            for(int i=0;i<3;++i) {
                design(3*i+axis,0)=3*a.hvp_directions_(3*i+axis,k)-sum;
                if(axis==0) design(3*i,1)=design(3*i,0);
            }
        }
        return design;
    };
    Eigen::Matrix2d normal=Eigen::Matrix2d::Zero();
    Eigen::Vector2d target=Eigen::Vector2d::Zero();
    for(const auto &a:atoms) {
        const double w=.7*a.fitness/std::pow(3.,1.3)/a.hvp_directions_.cols();
        for(int k=0;k<a.hvp_directions_.cols();++k) {
            auto design=rows(a,k);
            normal+=w*design.transpose()*design;
            target+=w*design.transpose()*a.hvp_targets_.col(k);
        }
    }
    normal*=mpi.size; target*=mpi.size;
    normal.diagonal().array()+=.23;
    const Eigen::Vector2d expected=normal.ldlt().solve(target);
    auto params=pair_hvp_settings(); params.Modify("hvp_pow","1.3"); params.Modify("weight_L2",".23");
    std::vector<Neighborhoods> nbhs;
    for(auto &a:atoms) nbhs.emplace_back(a,pot.cutoff);
    TrainerLinearPotential trainer(params); trainer.init(atoms,nbhs,&pot); trainer.Train();
    REQUIRE(pot.coefficients.isApprox(expected,1e-9));
    double squared_error=0.,loss=0.; int observations=0;
    for(const auto &a:atoms) {
        const double w=.7*a.fitness/std::pow(3.,1.3)/a.hvp_directions_.cols();
        for(int k=0;k<a.hvp_directions_.cols();++k) {
            const double error=(rows(a,k)*expected-a.hvp_targets_.col(k)).squaredNorm();
            squared_error+=error; loss+=w*error; observations+=9;
        }
    }
    REQUIRE(trainer.emonitor.hvp_rms==Approx(std::sqrt(squared_error/observations)).epsilon(1e-9));
    REQUIRE(trainer.emonitor.loss_hvp==Approx(loss/atoms.size()).epsilon(1e-9));
}

TEST_CASE("Linear HVP fitting combines EFS and scaled probes without changing input labels", "[train][linear][hvp]") {
    const double scale=GENERATE(1.,-2.5);
    PairQuadraticPotential pot; pot.coefficients<<1.7,-.4;
    std::vector<Atoms> atoms{pair_hvp_atom()};
    atoms[0].hvp_directions_*=scale; atoms[0].hvp_targets_*=scale;
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    pot.getEFS(atoms[0],nbhs[0]); const Atoms saved=atoms[0];
    pot.coefficients<<7.,-3.;
    auto params=pair_hvp_settings();
    params.Modify("energy_weight","1"); params.Modify("force_weight","1"); params.Modify("stress_weight","1");
    TrainerLinearPotential trainer(params); trainer.init(atoms,nbhs,&pot); trainer.Train();
    REQUIRE(pot.coefficients[0]==Approx(1.7).epsilon(1e-9));
    REQUIRE(pot.coefficients[1]==Approx(-.4).epsilon(1e-9));
    REQUIRE(trainer.emonitor.erms<1e-9);
    REQUIRE(trainer.emonitor.frms<1e-9);
    REQUIRE(trainer.emonitor.srms<1e-9);
    REQUIRE(trainer.emonitor.hvp_rms<1e-9);
    REQUIRE(atoms[0].pos_==saved.pos_);
    REQUIRE(atoms[0].forces_==saved.forces_);
    REQUIRE(atoms[0].hvp_targets_==saved.hvp_targets_);
}

TEST_CASE("Linear fitting ignores zero-weight HVP and rejects invalid HVP settings", "[train][linear][hvp]") {
    ConstantPotential pot; Data data;
    for(auto &a:data.atoms) {
        a.hvp_directions_=Eigen::MatrixXd::Ones(6,1);
        a.hvp_targets_=Eigen::MatrixXd::Constant(6,1,1e5);
    }
    auto params=settings();
    const int mode=GENERATE(0,1,2,3,4);
    if(mode==1) params.Modify("hvp_weight","-1");
    if(mode==2) params.Modify("hvp_fd_step","0");
    if(mode==3) params.Modify("hvp_pow","nan");
    if(mode==4) params.Modify("hvp_fd_step","inf");
    TrainerLinearPotential trainer(params); trainer.init(data.atoms,data.nbhs,&pot);
    if(mode) { REQUIRE_THROWS(trainer.Train()); REQUIRE(pot.coefficient==7); }
    else {
        trainer.Train(); REQUIRE(pot.coefficient==Approx(2));
        REQUIRE(trainer.emonitor.hvp_rms==0.); REQUIRE(trainer.emonitor.loss_hvp==0.);
    }
}

TEST_CASE("Linear HVP fitting rejects zero-curvature systems and synchronizes validation failures", "[train][linear][hvp]") {
    const bool invalid_validation=GENERATE(false,true);
    PairQuadraticPotential pot;
    std::vector<Atoms> atoms{pair_hvp_atom()};
    if(!invalid_validation) {
        // One atom has no neighbors, so every force-design row is exactly zero.
        atoms[0].resize(1); atoms[0].chemmap_={0};
        atoms[0].hvp_directions_=Eigen::MatrixXd::Ones(3,1);
        atoms[0].hvp_targets_=Eigen::MatrixXd::Zero(3,1);
    }
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    TrainerLinearPotential trainer(pair_hvp_settings()); trainer.init(atoms,nbhs,&pot);
    if(invalid_validation && mpi.rank==0) {
        auto validation=atoms; validation[0].hvp_targets_.resize(9,1);
        trainer.init_valid(validation,nbhs); trainer.monit_valid_err=true;
    }
    REQUIRE_THROWS(trainer.Train());
    if(!invalid_validation) REQUIRE(pot.coefficients==Eigen::Vector2d(7.,-3.));
}

TEST_CASE("Linear HVP fitting rebuilds neighborhoods for both displaced geometries", "[train][linear][hvp]") {
    PairQuadraticPotential pot; pot.cutoff=2.5; pot.cutoffsq=6.25; pot.coefficients<<1.7,-.4;
    std::vector<Atoms> atoms{pair_hvp_atom()};
    atoms[0].pos_[1]<<4.5,2.,2.; // A pair crosses the neighbor boundary along the probes.
    set_force_difference_targets(pot,atoms[0]);
    pot.coefficients<<7.,-3.;
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    TrainerLinearPotential trainer(pair_hvp_settings()); trainer.init(atoms,nbhs,&pot);
    trainer.Assemble(); trainer.Solve();
    REQUIRE(pot.coefficients[0]==Approx(1.7).epsilon(1e-7));
    REQUIRE(pot.coefficients[1]==Approx(-.4).epsilon(1e-7));
}

TEST_CASE("Linear descriptor HVP fits agree with independently differenced model forces", "[train][linear][hvp][descriptor]") {
    const std::string dtype="behler"; CAPTURE(dtype);
    GPSettings model;
    model.Modify("descriptor:dtype",dtype);
    model.Modify("descriptor:behler:cutoff","5");
    model.Modify("descriptor:behler:dispcount","2");
    model.Modify("descriptor:behler:dispselect","1 2");
    model.Modify("descriptor:soap:ntypes","1");
    model.Modify("descriptor:soap:nmax","2");
    model.Modify("descriptor:soap:lmax","1");
    model.Modify("descriptor:soap:rcut","4");
    model.Modify("descriptor:soap:sigma",".5");
    Potentials::PotentialDescriptorLinear pot(model);
    Eigen::VectorXd coefficients=Eigen::VectorXd::LinSpaced(pot.get_NL_coeffs_count(),-.2,.7);
    pot.nl_set_coeffs(coefficients.data());
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> nbhs;
    double squared_targets=0.; int observations=0;
    for(int i=0;i<8;++i) {
        auto a=pair_hvp_atom();
        a.pos_[1][0]+=.08*i; a.pos_[2][1]+=.04*i; a.pos_[2][2]+=.015*i*i;
        set_force_difference_targets(pot,a);
        squared_targets+=a.hvp_targets_.squaredNorm(); observations+=a.hvp_targets_.size();
        atoms.push_back(a); nbhs.emplace_back(atoms.back(),pot.cutoff);
    }
    coefficients.setZero(); pot.nl_set_coeffs(coefficients.data());
    auto params=pair_hvp_settings(); params.Modify("weight_L2","1e-12");
    TrainerLinearPotential trainer(params); trainer.init(atoms,nbhs,&pot);
    trainer.Assemble(); trainer.Solve();
    const double target_rms=std::sqrt(squared_targets/observations);
    REQUIRE(target_rms>0.);
    REQUIRE(trainer.emonitor.hvp_rms<1e-7*target_rms+1e-8);
}

#ifdef IMRMLP_MPI
TEST_CASE("Linear HVP fitting and validation permit empty MPI shards", "[train][linear][hvp][mpi]") {
    const int distribution=GENERATE(0,1,2);
    PairQuadraticPotential pot;
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> nbhs;
    if(distribution==1 || (distribution==0 && mpi.rank==0) ||
       (distribution==2 && (mpi.rank!=0 || mpi.size==1))) {
        atoms.push_back(pair_hvp_atom(1.7,-.4,mpi.rank+1));
        nbhs.emplace_back(atoms[0],pot.cutoff);
    }
    TrainerLinearPotential trainer(pair_hvp_settings()); trainer.init(atoms,nbhs,&pot);
    if(mpi.rank==0) {
        std::vector<Atoms> validation{pair_hvp_atom()};
        std::vector<Neighborhoods> valid_nbhs; valid_nbhs.emplace_back(validation[0],pot.cutoff);
        trainer.init_valid(validation,valid_nbhs); trainer.monit_valid_err=true;
    }
    trainer.Train();
    REQUIRE(pot.coefficients[0]==Approx(1.7).epsilon(1e-9));
    REQUIRE(trainer.emonitor.hvp_rms<1e-9);
    REQUIRE(trainer.emonitor_valid.hvp_rms<1e-9);
}
#endif
