#include <catch2/catch_all.hpp>
#include "mindriver/periodic_subspace.h"
#include "mindriver/dimer_new.h"
#include "mindriver/neb.h"
#include <Eigen/Geometry>
#include <filesystem>

namespace {
Atoms sample(int rank) {
    Atoms a;a.pbc_={false,false,false};a.cell_.setZero();
    const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.37,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    if(rank>0) {a.cell_.row(0)=Eigen::RowVector3d(7,0,0)*rotation;a.pbc_[0]=true;}
    if(rank>1) {a.cell_.row(2)=Eigen::RowVector3d(1,8,0)*rotation;a.pbc_[2]=true;}
    a.pos_={Eigen::RowVector3d(.4,.6,1.3)*rotation,Eigen::RowVector3d(1.2,.8,2.1)*rotation};
    a.chemmap_={0,0};return a;
}
void evaluate(Atoms &a) {
    a.energy_=0;a.stress_.setZero();a.forces_.resize(a.size());
    for(int i=0;i<a.size();++i) {
        a.energy_+=.5*a.pos_[i].squaredNorm();a.forces_[i]=-a.pos_[i].transpose();
        a.stress_-=a.pos_[i].transpose()*a.pos_[i];
    }
    for(int k=0;k<3;++k) if(a.pbc_[k]) {
        a.energy_+=.5*a.cell_.row(k).squaredNorm();a.stress_-=a.cell_.row(k).transpose()*a.cell_.row(k);
    }
}
struct SubspacePotential:Potentials::AnyPotential {
    SubspacePotential() {cutoff=2.;cutoffs=Eigen::MatrixXd::Constant(1,1,2.);cutoffsqs=Eigen::MatrixXd::Constant(1,1,4.);species_count=1;}
    void getEFS(Atoms &a,AnyNeighborhoods &) override {evaluate(a);}
    void getEFSOMP(Atoms &a,AnyNeighborhoods &) override {evaluate(a);}
    void getEFSOMPthread(Atoms &a) override {evaluate(a);}
};
}

TEST_CASE("Periodic Gram mixed coordinates have exact energy and geometry pullbacks", "[subspace][gradient]") {
    const int rank=GENERATE(0,1,2);const bool recast=GENERATE(false,true);
    auto a=sample(rank);PeriodicSubspaceCoordinates map;map.init(a,recast);
    REQUIRE(map.rank()==rank);REQUIRE(map.cell_dofs()==rank*(rank+1)/2);
    Eigen::VectorXd cell=Eigen::VectorXd::Constant(map.cell_dofs(),.08);
    a.cell_=map.decode_cell(cell);
    for(auto &p:a.pos_) p=map.decode_position(map.encode_position(p,a.cell_),a.cell_);
    evaluate(a);const auto cg=map.cell_gradient(a.cell_,a.stress_);
    const auto derivatives=map.cell_derivatives(a.cell_);
    for(double h:{1e-5,2e-6}) {
        for(int k=0;k<cell.size();++k) {
            auto plus=cell,minus=cell;plus[k]+=h;minus[k]-=h;
            auto ap=a,am=a;ap.cell_=map.decode_cell(plus);am.cell_=map.decode_cell(minus);
            for(int i=0;i<a.size();++i) {
                const auto q=map.encode_position(a.pos_[i],a.cell_);
                ap.pos_[i]=map.decode_position(q,ap.cell_);am.pos_[i]=map.decode_position(q,am.cell_);
            }
            evaluate(ap);evaluate(am);
            REQUIRE(cg[k]==Catch::Approx((ap.energy_-am.energy_)/(2*h)).epsilon(1e-7));
            REQUIRE(((ap.cell_-am.cell_)/(2*h)-derivatives[k]).norm()<1e-8);
            REQUIRE(ap.cell_.row(1).isZero());REQUIRE(am.cell_.row(1).isZero());
        }
        for(int i=0;i<a.size();++i) {
            const auto q=map.encode_position(a.pos_[i],a.cell_);
            REQUIRE((map.decode_position(q,a.cell_)-a.pos_[i]).norm()<1e-12);
            const Eigen::Vector3d gradient=-map.atomic_jacobian(a.cell_)*a.forces_[i];
            for(int d=0;d<3;++d) {
                auto qp=q,qm=q;qp[d]+=h;qm[d]-=h;auto ap=a,am=a;
                ap.pos_[i]=map.decode_position(qp,a.cell_);am.pos_[i]=map.decode_position(qm,a.cell_);
                evaluate(ap);evaluate(am);
                REQUIRE(gradient[d]==Catch::Approx((ap.energy_-am.energy_)/(2*h)).margin(1e-8));
            }
        }
    }
    auto b=a;const auto q=map.encode_position(a.pos_[0],a.cell_);
    if(rank) {
        const auto shifted=map.encode_position(a.pos_[0]+a.cell_.row(0),a.cell_);
        REQUIRE(map.displacement(q,shifted,a.cell_).norm()<1e-12);
    }
    b.cell_.row(1)<<0,0,9;
    REQUIRE_THROWS_WITH(map.encode_cell(b.cell_),Catch::Matchers::ContainsSubstring("Open cell directions"));
}

TEST_CASE("Dimer uses the shared periodic map for rank zero one and two and restarts", "[dimer][subspace]") {
    const int rank=GENERATE(0,1,2);const bool recast=GENERATE(false,true);
    auto a=sample(rank);DimerDriver driver;driver.vc_flag=true;driver.periodic_subspace=true;
    driver.recast_atomic_force=recast;driver.init(a);
    REQUIRE(driver.offset==rank*(rank+1)/2);REQUIRE(driver.nlen==driver.offset+6);
    evaluate(driver.atom_center);Eigen::VectorXd gradient;double energy;
    driver.LatPos2G(driver.atom_center,gradient,energy);
    for(double h:{1e-5,2e-6}) for(int k=0;k<driver.nlen;++k) {
        auto plus=driver.xx_center,minus=plus;plus[k]+=h;minus[k]-=h;auto ap=a,am=a;
        driver.X2LatPos(plus,ap);driver.X2LatPos(minus,am);evaluate(ap);evaluate(am);
        REQUIRE(gradient[k]==Catch::Approx((ap.energy_-am.energy_)/(2*h)).margin(2e-7));
        REQUIRE(ap.cartesian);REQUIRE(ap.cell_.row(1).isZero());
    }
    const auto filename=std::filesystem::temp_directory_path()/("imrmlp-subspace-mode-"+std::to_string(rank)+"-"+std::to_string(recast));
    driver.write_modecar(filename.string());DimerDriver restart;restart.vc_flag=true;restart.periodic_subspace=true;
    restart.recast_atomic_force=recast;restart.modecarf=filename.string();
    auto changed=a;auto moved=driver.xx_center;if(driver.offset) moved[0]+=.1;
    driver.X2LatPos(moved,changed);restart.init(changed);
    REQUIRE(restart.modedir.isApprox(driver.modedir,1e-14));
    REQUIRE(restart.refcell.isApprox(driver.refcell));
    auto decoded=changed;restart.X2LatPos(restart.xx_center,decoded);
    REQUIRE(decoded.cell_.isApprox(changed.cell_));
    REQUIRE((decoded.pos_[0]-changed.pos_[0]).norm()<1e-12);
    std::filesystem::remove(filename);
    driver.pressure=1;REQUIRE_THROWS(driver.init(a));
}

TEST_CASE("VC NEB shares mixed displacement interpolation gradient tangent and rebuilt images", "[neb][subspace]") {
    const int rank=GENERATE(0,1,2);SubspacePotential potential;auto first=sample(rank),last=first;
    PeriodicSubspaceCoordinates map;map.init(first);
    Eigen::VectorXd q=Eigen::VectorXd::Constant(map.cell_dofs(),.12);last.cell_=map.decode_cell(q);
    for(int i=0;i<last.size();++i) {
        auto x=map.encode_position(first.pos_[i],first.cell_);x[2]+=.3;
        last.pos_[i]=map.decode_position(x,last.cell_);
    }
    NEBDriver driver;driver.pot=&potential;driver.vcneb=true;driver.periodic_subspace=true;driver.iref=0;
    driver.init_method=GENERATE("linear","idpp");driver.idpp_max_steps=2;
    std::vector<Atoms> images{first,Atoms{},last};driver.init(images);
    REQUIRE(driver.vcoffset==map.cell_dofs());REQUIRE(driver.x.cols()==driver.vcoffset+6);
    for(const auto &image:driver.images) {REQUIRE(image.cartesian);REQUIRE(image.cell_.row(1).isZero());}
    Eigen::VectorXd x;driver.LatPos2X(x);Eigen::VectorXd g;driver.LatPos2X(x,g);
    REQUIRE(g.size()==x.size());REQUIRE(g.allFinite());
    g=Eigen::VectorXd::Constant(x.size()+4,std::numeric_limits<double>::quiet_NaN());
    driver.LatPos2X(x,g);REQUIRE(g.size()==x.size());REQUIRE(g.allFinite());
    REQUIRE(std::isfinite(driver.distance(driver.images.front(),driver.images.back(),0.)));
    const auto &middle=driver.images[1];const auto raw=map.cell_gradient(middle.cell_,middle.stress_);
    REQUIRE((driver.g.row(1).head(driver.vcoffset).transpose()+raw).norm()<1e-9);
    driver.iclimb_=0;driver.rebuild_path();REQUIRE(driver.images.size()==4);
    REQUIRE(driver.x.allFinite());
    for(const auto &image:driver.images) {REQUIRE(image.cartesian);REQUIRE(image.cell_.row(1).isZero());}
    driver.pressure=1;REQUIRE_THROWS(driver.init(images));
}
