#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include <numbers>

TEST_CASE("Mirrored FFT force reconstruction obeys mass and the Cartesian second derivative", "[fft][force]") {
    const bool even = GENERATE(false, true);
    const bool velocities = GENERATE(false, true);
    const int representation = GENERATE(0, 1, 2);
    const int count = 16, length = even ? 2*count : 2*count-1;
    const double dt = 2., omega = 2*std::numbers::pi/(length*dt);
    constexpr double conversion = 103.64269652680505;
    std::vector<Atoms> frames(count);
    for (int j=0;j<count;++j) {
        auto &a = frames[j];
        const double phase = omega*dt*(j+(even ? .5 : 0));
        a.cell_ = Eigen::Matrix3d::Identity()*(10+std::cos(phase));
        a.pos_ = {{2+.2*std::cos(phase),1,0},{1,3-.4*std::cos(2*phase),0}};
        a.chemmap_ = {0,1};
        a.local_attribs_d_["masses"] = Eigen::Vector2d(2.,5.);
        // Rebuilt forces replace old labels; later frames need not contain them.
        if (j == 0) a.forces_.assign(2,Eigen::RowVector3d::Constant(77.));
        if(representation==1 || (representation==2 && j%2)) a.to_fractional();
    }
    ComputeSmoothFFT smoother;
    smoother.even_seq=even; smoother.rebuild_force=true; smoother.rebuild_velo=velocities;
    smoother.params.freq_cutoff=1e6;
    smoother.compute2(frames,dt,ComputeSmoothFFT::BLOCK);
    for(int j=0;j<count;++j) {
        const double phase=omega*dt*(j+(even ? .5 : 0));
        REQUIRE(frames[j].forces_.size()==2);
        REQUIRE(frames[j].forces_[0][0]==Catch::Approx(-conversion*2*.2*omega*omega*std::cos(phase)).margin(1e-10));
        REQUIRE(frames[j].forces_[1][1]==Catch::Approx(conversion*5*.4*4*omega*omega*std::cos(2*phase)).margin(1e-10));
        REQUIRE(frames[j].forces_[0].tail<2>().norm()<1e-10);
        REQUIRE(std::abs(frames[j].forces_[1][0])<1e-10);
        REQUIRE(std::abs(frames[j].forces_[1][2])<1e-10);
        if(velocities) REQUIRE(frames[j].velocities_[0][0]==Catch::Approx(-.2*omega*std::sin(phase)).margin(1e-12));
        REQUIRE(frames[j].local_attribs_d_.at("masses").isApprox(Eigen::Vector2d(2.,5.)));
    }
}

TEST_CASE("FFT reconstruction requires physical masses before changing any frame", "[fft][force]") {
    const int invalid = GENERATE(0,1,2,3);
    std::vector<Atoms> frames(3);
    for(auto &a:frames) {
        a.pbc_={false,false,false}; a.cell_.setZero();
        a.pos_={{1,2,3}}; a.chemmap_={0}; a.local_attribs_d_["masses"]=Eigen::VectorXd::Constant(1,2.);
    }
    if(invalid==0) frames.front().local_attribs_d_.erase("masses");
    if(invalid==1) frames.back().local_attribs_d_["masses"][0]=0.;
    if(invalid==2) frames.back().local_attribs_d_["masses"][0]=3.;
    if(invalid==3) frames.back().local_attribs_d_.erase("masses");
    const auto before=frames;
    GPSettings settings;settings.Modify("rebuild_force","true");ComputeSmoothFFT smoother(settings);
    REQUIRE(smoother.rebuild_force);
    REQUIRE_THROWS(smoother.compute2(frames,1.,ComputeSmoothFFT::BLOCK));
    REQUIRE(frames.front().pos_==before.front().pos_);
    REQUIRE(frames.front().forces_.empty());
    REQUIRE_THROWS_WITH(smoother.compute(frames,1.,ComputeSmoothFFT::BLOCK),Catch::Matchers::ContainsSubstring("compute2"));
}

TEST_CASE("Two-frame force reconstruction matches an independent discrete Fourier sum", "[fft][force]") {
    // An independent DFT checks the entire two-frame mirror, including endpoints.
    const bool even=GENERATE(false,true);
    const int n=2, length=even ? 2*n : 2*n-1;
    std::vector<Atoms> frames(n);
    for(int j=0;j<n;++j) {
        frames[j].pbc_={false,false,false};frames[j].cell_.setZero();
        frames[j].pos_={{double(j),0,0}};frames[j].chemmap_={0};
        frames[j].local_attribs_d_["masses"]=Eigen::VectorXd::Ones(1);
    }
    ComputeSmoothFFT s;s.even_seq=even;s.rebuild_force=true;s.params.freq_cutoff=1e6;
    s.compute2(frames,1.,ComputeSmoothFFT::BLOCK);
    // Independently sum the DFT second derivative of the exact mirror samples.
    const std::vector<double> mirrored=even ? std::vector<double>{0,1,1,0} : std::vector<double>{0,1,1};
    for(int j=0;j<n;++j) {
        double acceleration=0;
        for(int q=0;q<length;++q) {
            const int k=q<=length/2 ? q : q-length;
            double real=0,imag=0;
            for(int l=0;l<length;++l) {
                const double phase=2*std::numbers::pi*q*l/length;
                real+=mirrored[l]*std::cos(phase);imag-=mirrored[l]*std::sin(phase);
            }
            const double phase=2*std::numbers::pi*q*j/length;
            acceleration-=std::pow(2*std::numbers::pi*k/length,2)*(real*std::cos(phase)-imag*std::sin(phase))/length;
        }
        REQUIRE(frames[j].forces_[0][0]==Catch::Approx(103.64269652680505*acceleration).margin(1e-10));
    }
}
