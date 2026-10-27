#pragma once
#include <catch2/catch_all.hpp>
#include "unitest/pair_cell_fixture.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
namespace EAMSplineTest {
struct Fixture {
    std::filesystem::path directory;std::string file;
    Fixture(bool fs,int mode){char name[]="/tmp/imrmlp_eam_curve_XXXXXX";const char *p=mkdtemp(name);if(!p)throw std::runtime_error("mkdtemp failed");directory=p;file=(directory/"model.table").string();
        const int count=mode==1?17:mode==2?5:33;const double step=mode==2?.1:1.,origin=mode==1?-8.:0.;
        std::ofstream out(file);out<<std::setprecision(17)<<"a\nb\nc\n2 H He\n"<<count<<' '<<step<<" 5 1 5"<<(mode==1?" 8":"")<<'\n';
        for(int neighbor=0;neighbor<2;++neighbor){out<<neighbor+1<<" 1 1 fcc\n";
            for(int k=0;k<count;++k){const double rho=origin+k*step;out<<(.15+.05*neighbor)*rho*rho<<' ';}out<<'\n';
            for(int host=0;host<(fs?2:1);++host){for(int r=0;r<5;++r)out<<(mode==1?-1:1)*(.08+.03*neighbor+.05*host)*r<<' ';out<<'\n';}}
        for(int pair=0;pair<3;++pair){for(int r=0;r<5;++r)out<<(.025+.015*pair)*r*r*(1+.05*r)<<' ';out<<'\n';}}
    ~Fixture(){std::filesystem::remove_all(directory);}
    template<class Model>std::unique_ptr<Model> make(bool reverse=false){std::map<std::string,int> map;if(reverse)map={{"H",1},{"He",0}};return std::make_unique<Model>(file,map);}
};
inline Eigen::MatrixXd force_fd(Potentials::AnyPotential &pot,const Atoms &a,double h){Eigen::MatrixXd result(3*a.size(),3*a.size());
    for(int k=0;k<result.cols();++k){auto plus=a,minus=a;plus.pos_[k/3][k%3]+=h;minus.pos_[k/3][k%3]-=h;pot.getEFS(plus);pot.getEFS(minus);
        for(int i=0;i<a.size();++i)result.block<3,1>(3*i,k)=-(plus.forces_[i]-minus.forces_[i]).transpose()/(2*h);}return result;}
inline void close(const Eigen::MatrixXd &a,const Eigen::MatrixXd &b,double h){const double scale=std::max(1.,b.norm());INFO("error="<<(a-b).norm()<<" scale="<<scale);REQUIRE((a-b).norm()<80*h*h*scale+2e-6);}
template<class Model>void native(bool fs,int mode,bool partial){Fixture files(fs,mode);auto owner=files.make<Model>();auto &api=*owner;auto a=PairCellTest::geometry(partial);a.pos_[1][1]+=.071;auto expected=a;api.getEFS(expected);
    Potentials::HessianMatrix blocks;Eigen::MatrixXd cell,cross;Neighborhoods nbhs(a,api.cutoff);api.getEFSHessianCellOMP(a,nbhs,blocks,cell,cross);const auto full=PairCellTest::native_matrix(blocks,cell,cross);
    REQUIRE(cell.norm()>1e-6);REQUIRE(cross.norm()>1e-6);
    for(double h:{.001,.0005})close(full,PairCellTest::energy_fd(api,a,h,true),h);
    const auto atomic=HessianUtils::to_dense(blocks);for(double h:{1e-5,5e-6})REQUIRE((atomic-force_fd(api,a,h)).norm()<1e-7*std::max(1.,atomic.norm())+2e-7);
    REQUIRE(a.energy_==expected.energy_);REQUIRE(a.forces_==expected.forces_);REQUIRE(a.stress_==expected.stress_);
    api.add_efs=true;api.getEFSHessianCellSymmOMP(a,blocks,cell,cross);REQUIRE((full-PairCellTest::native_matrix(blocks,cell,cross)).norm()<1e-10);
    REQUIRE(std::abs(a.energy_-2*expected.energy_)<1e-10);REQUIRE((a.stress_-2*expected.stress_).norm()<1e-10);for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*expected.forces_[i]).norm()<1e-10);
}
template<class Model>void consumer(bool fs,bool strain,double cutoff){Fixture files(fs,0);auto owner=files.make<Model>();auto &api=*owner;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&api);for(double h:{.001,.0005})close(result,PairCellTest::energy_fd(api,a,h,false,strain,cutoff),h);
    c.request_response_pos_cell=false;c.compute(a,result,&api);REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
template<class Model>void self(bool fs,int mode){Fixture files(fs,mode);auto owner=files.make<Model>();auto &api=*owner;Atoms a;a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.2,.3,.4}};a.chemmap_={0};
    Potentials::HessianMatrix blocks;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymm(a,blocks,cell,cross);REQUIRE(cell.norm()>1e-6);REQUIRE(blocks(0,0).isZero(1e-11));REQUIRE(cross.isZero(1e-11));
    for(double h:{.001,.0005})close(PairCellTest::native_matrix(blocks,cell,cross),PairCellTest::energy_fd(api,a,h,true),h);
    a.cell_.setZero();REQUIRE_THROWS(api.getEFSHessianCellSymmOMP(a,blocks,cell,cross));
}
template<class Model>void mapping(bool fs){Fixture files(fs,0);auto original=files.make<Model>(),mapped=files.make<Model>(true);auto a=PairCellTest::geometry(),b=a;b.chemmap_={1,0,1};
    Potentials::HessianMatrix ha,hb;Eigen::MatrixXd ca,cb,xa,xb;original->getEFSHessianCellSymm(a,ha,ca,xa);mapped->getEFSHessianCellSymm(b,hb,cb,xb);
    REQUIRE((PairCellTest::native_matrix(ha,ca,xa)-PairCellTest::native_matrix(hb,cb,xb)).norm()<1e-10);REQUIRE(std::abs(a.energy_-b.energy_)<1e-11);
}
}
