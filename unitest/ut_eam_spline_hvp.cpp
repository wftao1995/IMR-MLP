#include "potentials/potential_eamalloy_spline.h"
#include "potentials/potential_eamfs_spline.h"
#include "eam_spline_curvature_fixture.h"
#include <limits>
#include "apps/apps.h"
#include "io/gpio.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
using namespace Potentials;
namespace {
Eigen::VectorXd force_direction_fd(AnyPotential &api,const Atoms &a,const Eigen::VectorXd &v,double h,double cutoff){
    auto plus=a,minus=a;for(int i=0;i<a.size();++i){plus.pos_[i]+=h*v.segment<3>(3*i).transpose();minus.pos_[i]-=h*v.segment<3>(3*i).transpose();}
    Neighborhoods np(plus,cutoff),nm(minus,cutoff);api.getEFS(plus,np);api.getEFS(minus,nm);
    Eigen::VectorXd result(v.size());for(int i=0;i<a.size();++i)result.segment<3>(3*i)=-(plus.forces_[i]-minus.forces_[i]).transpose()/(2*h);return result;
}
template<class Model>void check(bool fs,int mode,bool self,double cutoff){
    EAMSplineTest::Fixture files(fs,mode);auto owner=files.make<Model>();auto &api=*owner;
    auto a=PairCellTest::geometry();a.pos_[1][1]+=.071;
    if(self){a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.2,.3,.4}};a.chemmap_={0};}
    const double used_cutoff=cutoff?cutoff:api.cutoff;Neighborhoods nbhs(a,used_cutoff);api.getEFS(a,nbhs);
    const auto energy=a.energy_;const auto forces=a.forces_;const auto stress=a.stress_;
    REQUIRE(api.supports_analytical_hessian_vector_product());
    HessianMatrix blocks;auto fresh=a;api.getEFSHessian(fresh,nbhs,blocks);const auto dense=HessianUtils::to_dense(blocks);
    for(int probe=0;probe<3;++probe){Eigen::VectorXd v(3*a.size());for(int i=0;i<v.size();++i)v[i]=std::sin(.71*(i+1)*(probe+1));
        Eigen::VectorXd product=Eigen::VectorXd::Constant(2,123.);api.applyHessianVectorProduct(a,nbhs,v,product);
        REQUIRE((product-dense*v).norm()<1e-11*std::max(1.,product.norm()));
        for(double h:{1e-5,5e-6}){const auto fd=force_direction_fd(api,a,v,h,used_cutoff);INFO("mode="<<mode<<" fs="<<fs<<" self="<<self<<" cutoff="<<cutoff);REQUIRE((product-fd).norm()<2e-7*std::max(1.,product.norm()));}
        auto alias=v;api.applyHessianVectorProduct(a,nbhs,alias,alias);REQUIRE((alias-product).norm()<1e-12);
        api.add_efs=true;Eigen::VectorXd additive;api.applyHessianVectorProduct(a,nbhs,v,additive);api.add_efs=false;REQUIRE(additive==product);
        ComputeHessian consumer;consumer.method=ComputeHessian::HessianMethod::Analytic;consumer.set_cutoff(cutoff);
        Eigen::VectorXd consumed;consumer.compute_hvp(a,v,consumed,&api);REQUIRE((consumed-product).norm()<1e-12);
        auto fractional=a;fractional.to_fractional();consumer.compute_hvp(fractional,v,consumed,&api);REQUIRE((consumed-product).norm()<1e-11);
        consumer.method=ComputeHessian::HessianMethod::Auto;consumer.compute_hvp(a,v,consumed,&api);REQUIRE((consumed-product).norm()<1e-12);
        consumer.fractional_pos=true;REQUIRE_THROWS(consumer.compute_hvp(a,v,consumed,&api));
    }
    Eigen::VectorXd translation=Eigen::VectorXd::Ones(3*a.size()),zero;api.applyHessianVectorProduct(a,nbhs,translation,zero);REQUIRE(zero.isZero(1e-12));
    REQUIRE(a.energy_==energy);REQUIRE(a.forces_==forces);REQUIRE(a.stress_==stress);
    Eigen::VectorXd bad=Eigen::VectorXd::Zero(3*a.size()+1),product;REQUIRE_THROWS(api.applyHessianVectorProduct(a,nbhs,bad,product));
    bad=Eigen::VectorXd::Zero(3*a.size());bad[0]=std::numeric_limits<double>::quiet_NaN();REQUIRE_THROWS(api.applyHessianVectorProduct(a,nbhs,bad,product));
    auto other=a;other.pos_.push_back({.1,.1,.1});other.chemmap_.push_back(0);Neighborhoods wrong(other,used_cutoff);REQUIRE_THROWS(api.applyHessianVectorProduct(a,wrong,translation,product));
}
}
TEST_CASE("Spline EAM matrix-free HVP contracts actual site curvature","[eam][hvp][batch47]"){
    const bool fs=GENERATE(false,true);const int mode=GENERATE(0,1,2);const double cutoff=GENERATE(0.,3.1,4.7);
    if(fs)check<EAMFsSpline>(true,mode,false,cutoff);else check<EAMAlloySpline>(false,mode,false,cutoff);
}
TEST_CASE("Spline EAM HVP cancels repeated self source while retaining embedding density","[eam][hvp][batch47]"){
    const bool fs=GENERATE(false,true);const int mode=GENERATE(0,1,2);
    if(fs)check<EAMFsSpline>(true,mode,true,0.);else check<EAMAlloySpline>(false,mode,true,0.);
}

TEST_CASE("Ordinary spline EAM native HVP drives calc_fc lowest modes","[eam][hvp][cli][batch47]"){
    const bool fs=GENERATE(false,true);
    EAMSplineTest::Fixture files(fs,0);
    struct Workspace {
        CMDOption::CommandLineOptions saved=CMDOption::global_cmdopts;
        std::filesystem::path original=std::filesystem::current_path();int threads=omp_num_threads;
        explicit Workspace(const std::filesystem::path &p){std::filesystem::current_path(p);omp_num_threads=1;global_omp_init();}
        ~Workspace(){CMDOption::global_cmdopts=saved;omp_num_threads=threads;global_omp_init();std::filesystem::current_path(original);}
    } work(files.directory);
    std::unique_ptr<AnyPotential> model;
    if(fs)model=files.make<EAMFsSpline>();else model=files.make<EAMAlloySpline>();
    auto a=PairCellTest::geometry();a.pos_[1][1]+=.071;const int dimension=3*a.size();
    Eigen::MatrixXd fd(dimension,dimension);
    for(int k=0;k<dimension;++k)fd.col(k)=force_direction_fd(*model,a,Eigen::VectorXd::Unit(dimension,k),5e-6,model->cutoff);
    REQUIRE((fd-fd.transpose()).norm()<1e-6);fd=(.5*(fd+fd.transpose())).eval();
    Eigen::VectorXd roots(dimension);Eigen::MatrixXd translations=Eigen::MatrixXd::Zero(dimension,3);
    for(int i=0;i<a.size();++i){const double root=std::sqrt(a.chemmap_[i]==0?2.:5.);roots.segment<3>(3*i).setConstant(root);for(int axis=0;axis<3;++axis)translations(3*i+axis,axis)=root;}
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
    const Eigen::MatrixXd q=qr.householderQ()*Eigen::MatrixXd::Identity(dimension,dimension),internal=q.rightCols(dimension-3);
    const Eigen::MatrixXd dynamical=roots.cwiseInverse().asDiagonal()*fd*roots.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(internal.transpose()*dynamical*internal);REQUIRE(solver.info()==Eigen::Success);
    std::ofstream("index.map")<<"H 1.0 2\nHe 1.0 5\n";GPIO io;REQUIRE(io.readmap("index.map"));REQUIRE(io.write_one(a,"input.xyz","extxyz"));
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions{};
    std::vector<std::string> args{"calc_fc","--potfile="+files.file,"--potential-type="+std::string(fs?"eamfs":"eamalloy"),"--input=input.xyz","--input-format=extxyz","--index=index.map","--hessian-method=analytic","--lowest-modes=2","--lowest-mode-tol=1e-9"};
    std::vector<char*> argv;for(auto &arg:args)argv.push_back(arg.data());REQUIRE(calc_fc(argv.size(),argv.data())==0);REQUIRE_FALSE(std::filesystem::exists("FORCE_CONSTANTS"));
    std::ifstream output("EIGENVALS");std::string header;REQUIRE(std::getline(output,header));for(int mode=0;mode<2;++mode){double eigenvalue,frequency,energy,wavenumber;REQUIRE(output>>eigenvalue>>frequency>>energy>>wavenumber);REQUIRE(eigenvalue==Catch::Approx(solver.eigenvalues()[mode]).margin(2e-6));}
}
