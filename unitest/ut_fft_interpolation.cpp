#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include "analyse/fft_resample.h"

namespace {
std::vector<Atoms> harmonic_frames(bool even,bool periodic) {
    const int n=12,length=2*n-(even?0:1);std::vector<Atoms> frames(n);
    for(int j=0;j<n;++j) {
        auto& f=frames[j];const double phase=2*std::acos(-1.)*(j+(even?.5:0.))/length;
        f.cell_=periodic ? Eigen::Matrix3d(Eigen::Matrix3d::Identity()*(8+.1*std::cos(phase))) : Eigen::Matrix3d::Zero();
        f.pbc_={periodic,periodic,periodic};f.pos_={{2+.2*std::cos(phase),1+.3*std::cos(2*phase),.5}};f.chemmap_={0};
        f.energy_=3+.4*std::cos(phase);f.fenergy_=2+.1*std::cos(phase);f.have_energy=f.have_fenergy=f.have_stress=true;
        f.stress_=Eigen::Matrix3d::Constant(.3+.2*std::cos(phase));
        f.site_energies_={f.energy_};f.forces_={Eigen::Vector3d::Constant(5+.1*std::cos(phase))};
        f.velocities_={Eigen::Vector3d::Constant(7+.1*std::cos(phase))};
        f.local_attribs_d_["masses"]=Eigen::VectorXd::Constant(1,4.);
        f.local_attribs_d_["scalar"]=Eigen::VectorXd::Constant(1,2+std::cos(phase));
        f.local_attribs_v_["vector"]=Eigen::MatrixXd::Constant(2,1,4+std::cos(phase));
        f.global_attribs_d_["scalar"]=6+std::cos(phase);f.global_attribs_v_["vector"]=Eigen::Vector2d(1+std::cos(phase),2+std::cos(phase));
        f.global_attribs_i_["step"]=100+j;f.local_attribs_i_["kind"]=Eigen::VectorXi::Constant(1,j);
        f.features_["text"]=std::to_string(j);f.id=100+j;
    }
    return frames;
}
}
TEST_CASE("FFT interpolation preserves original samples and Fourier Cartesian derivatives","[fft][interpolation]") {
    const bool even=GENERATE(false,true),periodic=GENERATE(false,true);
    const int q=GENERATE(2,3);const bool rebuild=GENERATE(false,true);
    auto frames=harmonic_frames(even,periodic);
    if(periodic) for(int j=1;j<int(frames.size());j+=2) frames[j].to_fractional();
    auto reference=frames;
    if(rebuild) {frames.back().forces_.clear();frames.back().velocities_.clear();reference=frames;}
    ComputeSmoothFFT old,interpolated;old.even_seq=interpolated.even_seq=even;
    old.rebuild_force=interpolated.rebuild_force=rebuild;old.rebuild_velo=interpolated.rebuild_velo=rebuild;
    old.params.freq_cutoff=interpolated.params.freq_cutoff=1e6;interpolated.interpolation_factor=q;
    const double dt=2.;old.compute2(reference,dt,ComputeSmoothFFT::BLOCK);interpolated.compute2(frames,dt,ComputeSmoothFFT::BLOCK);
    REQUIRE(frames.size()==size_t(q*(reference.size()-1)+1));
    const double omega=2*std::acos(-1.)/((2*reference.size()-(even?0:1))*dt);
    for(int m=0;m<int(frames.size());++m) {
        const auto& f=frames[m];const double phase=omega*dt*(double(m)/q+(even?.5:0.));
        REQUIRE(f.pos_[0][0]==Catch::Approx(2+.2*std::cos(phase)).margin(2e-12));
        REQUIRE(f.pos_[0][1]==Catch::Approx(1+.3*std::cos(2*phase)).margin(2e-12));
        REQUIRE(f.energy_==Catch::Approx(3+.4*std::cos(phase)).margin(2e-12));
        REQUIRE(f.fenergy_==Catch::Approx(2+.1*std::cos(phase)).margin(2e-12));
        REQUIRE(f.stress_(1,2)==Catch::Approx(.3+.2*std::cos(phase)).margin(2e-12));
        REQUIRE(f.site_energies_[0]==Catch::Approx(f.energy_).margin(2e-12));
        REQUIRE(f.local_attribs_d_.at("scalar")[0]==Catch::Approx(2+std::cos(phase)).margin(2e-12));
        REQUIRE(f.local_attribs_v_.at("vector")(1,0)==Catch::Approx(4+std::cos(phase)).margin(2e-12));
        REQUIRE(f.global_attribs_v_.at("vector")[1]==Catch::Approx(2+std::cos(phase)).margin(2e-12));
        REQUIRE(f.global_attribs_d_.at("fft_elapsed_fs")==Catch::Approx(m*dt/q));REQUIRE(f.id==m);
        REQUIRE(f.features_.at("text")==std::to_string(m/q));REQUIRE(f.global_attribs_i_.at("step")==100+m/q);
        REQUIRE(f.local_attribs_i_.at("kind")[0]==m/q);REQUIRE(f.cartesian);
        if(rebuild) {
            REQUIRE(f.velocities_[0][0]==Catch::Approx(-.2*omega*std::sin(phase)).margin(2e-12));
            REQUIRE(f.forces_[0][0]==Catch::Approx(-103.64269652680505*4*.2*omega*omega*std::cos(phase)).margin(2e-10));
            REQUIRE(f.local_attribs_d_.at("masses")[0]==4.);
        } else {
            REQUIRE(f.velocities_[0][0]==Catch::Approx(7+.1*std::cos(phase)).margin(2e-12));
            REQUIRE(f.forces_[0][0]==Catch::Approx(5+.1*std::cos(phase)).margin(2e-12));
        }
        if(m%q==0) {
            const auto& r=reference[m/q];REQUIRE((f.cell_-r.cell_).norm()<1e-11);
            REQUIRE((f.pos_[0]-r.pos_[0]).norm()<1e-11);REQUIRE((f.forces_[0]-r.forces_[0]).norm()<1e-9);
            REQUIRE((f.velocities_[0]-r.velocities_[0]).norm()<1e-10);
        }
    }
}
TEST_CASE("FFT interpolated frames discard unsupported calculation labels","[fft][interpolation][labels]") {
    auto frames=harmonic_frames(true,false);
    for(auto& f:frames) {
        f.have_hessian=true;f.hessian_=Eigen::Matrix3d::Identity();
        f.hvp_directions_=f.hvp_targets_=Eigen::MatrixXd::Ones(3,1);
        f.charges_={.5};f.magmom_scalar_={1.};f.magmoms_={Eigen::Vector3d::Ones()};
        f.have_eff_chg=true;f.born_eff_chgs_={Eigen::Matrix3d::Identity()};
        f.features_["hessian"]="stale";f.local_attribs_d_["charges"]=Eigen::VectorXd::Ones(1);
        f.global_attribs_d_["total_charge"]=2.;
        f.features_["fft_elapsed_fs"]="previous run";f.global_attribs_i_["fft_elapsed_fs"]=123;
    }
    ComputeSmoothFFT smoother;smoother.interpolation_factor=2;smoother.params.freq_cutoff=1e6;
    smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK);
    for(const auto& f:frames) {
        REQUIRE_FALSE(f.has_hessian());REQUIRE_FALSE(f.has_hvp());REQUIRE_FALSE(f.has_charges());
        REQUIRE_FALSE(f.has_magmom());REQUIRE_FALSE(f.has_eff_charges());REQUIRE_FALSE(f.have_eff_chg);
        REQUIRE_FALSE(f.features_.contains("hessian"));REQUIRE_FALSE(f.local_attribs_d_.contains("charges"));
        REQUIRE(f.global_attribs_d_.at("total_charge")==Catch::Approx(2.));
        REQUIRE_FALSE(f.features_.contains("fft_elapsed_fs"));REQUIRE_FALSE(f.global_attribs_i_.contains("fft_elapsed_fs"));
    }
}
TEST_CASE("FFT interpolation failures never publish partial geometry","[fft][interpolation][errors]") {
    auto frames=harmonic_frames(true,false);const auto original=frames;
    ComputeSmoothFFT smoother;smoother.interpolation_factor=0;
    REQUIRE_THROWS(smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK));REQUIRE(frames.size()==original.size());
    smoother.interpolation_factor=2;smoother.rebuild_force=true;frames.back().local_attribs_d_.erase("masses");
    REQUIRE_THROWS(smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK));
    for(int j=0;j<int(frames.size());++j) REQUIRE(frames[j].pos_==original[j].pos_);
    frames=harmonic_frames(true,true);frames.resize(4);
    for(int j=0;j<4;++j) {frames[j].cell_=Eigen::Matrix3d::Identity()*(j==3?100.:1.);frames[j].pos_[0].setZero();}
    const auto cells=frames;smoother.rebuild_force=false;smoother.params.freq_cutoff=1e6;
    REQUIRE_THROWS_WITH(smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK),Catch::Matchers::ContainsSubstring("cell"));
    REQUIRE(frames.size()==cells.size());for(int j=0;j<4;++j) REQUIRE(frames[j].cell_==cells[j].cell_);
}

TEST_CASE("FFT interpolation retains fixed singular partial cells and open coordinates","[fft][interpolation][partial-cell]") {
    auto frames=harmonic_frames(true,false);
    for(auto& f:frames) {f.cell_.setZero();f.cell_(0,0)=8.;f.cell_(2,2)=8.;f.pbc_={true,false,true};}
    ComputeSmoothFFT smoother;smoother.interpolation_factor=3;smoother.params.freq_cutoff=1e6;
    smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK);
    REQUIRE(frames.size()==34);
    for(int m=0;m<int(frames.size());++m) {
        const double phase=2*std::acos(-1.)*(double(m)/3+.5)/24;
        REQUIRE(frames[m].cell_.row(1).isZero());REQUIRE(frames[m].pbc_[1]==false);
        REQUIRE(frames[m].pos_[0][1]==Catch::Approx(1+.3*std::cos(2*phase)).margin(2e-12));
    }
}

TEST_CASE("FFT triclinic partial PBC interpolation preserves wrapped cell motion derivatives","[fft][interpolation][triclinic]") {
    const bool even=GENERATE(false,true);const int q=GENERATE(2,3);
    const int n=12,length=2*n-(even?0:1);const double dt=2.,omega=2*std::acos(-1.)/(length*dt);
    Eigen::Matrix3d base,modulation;
    base<<8.,.3,.2,.2,9.,.4,.1,.2,10.;
    modulation<<.1,.02,0.,-.01,.05,.01,0.,.01,.08;
    std::vector<Atoms> frames(n);
    for(int j=0;j<n;++j) {
        const double phase=omega*dt*(j+(even?.5:0.));auto& f=frames[j];
        f.cell_=base+std::cos(phase)*modulation;f.pbc_={true,false,true};
        const Eigen::RowVector3d fractional(.95+.15*std::cos(phase),.3+.05*std::cos(2*phase),.4+.03*std::cos(phase));
        f.pos_={fractional*f.cell_};f.chemmap_={0};f.local_attribs_d_["masses"]=Eigen::VectorXd::Constant(1,2.);
        f.wrap();if(j%2) f.to_fractional();
    }
    auto reference=frames;ComputeSmoothFFT old,smoother;
    old.even_seq=smoother.even_seq=even;old.params.freq_cutoff=smoother.params.freq_cutoff=1e6;
    smoother.interpolation_factor=q;smoother.rebuild_force=smoother.rebuild_velo=true;
    old.compute2(reference,dt,ComputeSmoothFFT::BLOCK);smoother.compute2(frames,dt,ComputeSmoothFFT::BLOCK);
    for(int m=0;m<int(frames.size());++m) {
        const double phase=omega*dt*(double(m)/q+(even?.5:0.));
        const auto cell=(base+std::cos(phase)*modulation).eval();
        const auto cell_first=(-omega*std::sin(phase)*modulation).eval();
        const auto cell_second=(-omega*omega*std::cos(phase)*modulation).eval();
        // First wrapped sample fixes the origin branch: f_x is shifted by -1.
        const Eigen::RowVector3d fractional(-.05+.15*std::cos(phase),.3+.05*std::cos(2*phase),.4+.03*std::cos(phase));
        const Eigen::RowVector3d first(-.15*omega*std::sin(phase),-.1*omega*std::sin(2*phase),-.03*omega*std::sin(phase));
        const Eigen::RowVector3d second(-.15*omega*omega*std::cos(phase),-.2*omega*omega*std::cos(2*phase),-.03*omega*omega*std::cos(phase));
        const auto velocity=(first*cell+fractional*cell_first).eval();
        const auto acceleration=(second*cell+2*first*cell_first+fractional*cell_second).eval();
        REQUIRE((frames[m].cell_-cell).norm()<2e-11);
        REQUIRE((frames[m].pos_[0]-fractional*cell).norm()<2e-11);
        REQUIRE((frames[m].velocities_[0]-velocity.transpose()).norm()<2e-11);
        REQUIRE((frames[m].forces_[0]-2*103.64269652680505*acceleration.transpose()).norm()<2e-9);
        if(m%q==0) REQUIRE((frames[m].pos_[0]-reference[m/q].pos_[0]).norm()<2e-11);
    }
}

TEST_CASE("FFT interpolation retains old-grid samples after an actual low-pass window","[fft][interpolation][compatibility]") {
    const bool even=GENERATE(false,true);auto frames=harmonic_frames(even,true);
    const int n=frames.size(),length=2*n-(even?0:1);
    for(int j=0;j<n;++j) frames[j].pos_[0][0]+=.05*std::cos(12*std::acos(-1.)*(j+(even?.5:0.))/length);
    const auto source=frames;auto reference=frames;
    ComputeSmoothFFT old,smoother;old.even_seq=smoother.even_seq=even;
    old.params.freq_cutoff=smoother.params.freq_cutoff=.16;
    old.rebuild_force=smoother.rebuild_force=true;old.rebuild_velo=smoother.rebuild_velo=true;
    smoother.interpolation_factor=3;
    old.compute2(reference,2.,ComputeSmoothFFT::BLOCK);smoother.compute2(frames,2.,ComputeSmoothFFT::BLOCK);
    double change=0.;
    for(int j=0;j<n;++j) {
        change=std::max(change,(reference[j].pos_[0]-source[j].pos_[0]).norm());
        REQUIRE((frames[3*j].pos_[0]-reference[j].pos_[0]).norm()<2e-11);
        REQUIRE((frames[3*j].cell_-reference[j].cell_).norm()<2e-11);
        REQUIRE((frames[3*j].forces_[0]-reference[j].forces_[0]).norm()<2e-9);
        REQUIRE((frames[3*j].velocities_[0]-reference[j].velocities_[0]).norm()<2e-11);
        REQUIRE(frames[3*j].energy_==Catch::Approx(reference[j].energy_).margin(2e-11));
        REQUIRE((frames[3*j].stress_-reference[j].stress_).norm()<2e-11);
    }
    REQUIRE(change>1e-3);
}
