#include <catch2/catch_all.hpp>
#include "mindriver/neb_perturbation.h"
#include "mindriver/neb_symmetry.h"

namespace {
std::vector<Atoms> seed_band(bool symmetric=true) {
    Atoms a;a.cell_=Eigen::Vector3d(6,7,8).asDiagonal();a.chemmap_={0,0};
    a.pos_={Eigen::RowVector3d(1.2,1.3,1.4),Eigen::RowVector3d(4.8,1.3,1.4)};
    if(!symmetric)a.pbc_={true,true,false};
    std::vector<Atoms> path(7,a);
    for(int i=0;i<7;++i){path[i].pos_[0][0]+=.03*i;path[i].pos_[1][0]-=.03*i;}
    return path;
}
}
TEST_CASE("Path seeds need no potential and preserve source endpoint labels and bounds", "[neb][seeds]") {
    auto path=seed_band();NEBDriver d;d.prepare_complete_path(path,path[0].cell_);const auto x=d.x;
    for(const std::string mode:{"sine1","sine2","mixed3"}) {
        NEBSeedOptions o;o.mode=mode;o.seed=19;const auto a=NEBPathSeeds::generate(d,o),b=NEBPathSeeds::generate(d,o);
        REQUIRE(a.status=="generated");REQUIRE(a.atomic_displacement<=.01000000001);REQUIRE(a.atomic_displacement>.0099);
        REQUIRE(a.subspace=="symmetry_complement");REQUIRE(a.images.front().pos_==path.front().pos_);REQUIRE(a.images.back().pos_==path.back().pos_);
        for(int i=0;i<7;++i)REQUIRE(a.images[i].pos_==b.images[i].pos_);
        o.sign=-1;auto minus=NEBPathSeeds::generate(d,o);
        for(int i=1;i<6;++i)for(int j=0;j<2;++j) REQUIRE((a.images[i].pos_[j]+minus.images[i].pos_[j]-2*path[i].pos_[j]).norm()<1e-12);
    }
    REQUIRE(d.x==x);REQUIRE(d.image_force_evaluations==0);
    REQUIRE(d.x.row(0).head<3>().isApprox(path[0].pos_[0]));
    REQUIRE(d.x.allFinite());
    for(int i=0;i<7;++i)REQUIRE(d.images[i].pos_==path[i].pos_);
}
TEST_CASE("Independent smooth modes differ and unsupported zero modes are skipped", "[neb][seeds]") {
    auto path=seed_band();NEBDriver d;d.prepare_complete_path(path,path[0].cell_);NEBSeedOptions o;
    auto first=NEBPathSeeds::generate(d,o);o.mode="sine2";auto second=NEBPathSeeds::generate(d,o);
    REQUIRE((second.images[3].pos_[0]-path[3].pos_[0]).norm()<1e-12);
    REQUIRE((first.images[3].pos_[0]-path[3].pos_[0]).norm()>1e-4);
    o.mode="mixed3";auto mixed=NEBPathSeeds::generate(d,o);
    REQUIRE((mixed.images[2].pos_[0]-first.images[2].pos_[0]).norm()>1e-5);
    path={path[0],path[3],path[6]};d.prepare_complete_path(path,path[0].cell_);o.mode="sine2";
    REQUIRE(NEBPathSeeds::generate(d,o).status=="skipped");
}
TEST_CASE("Local seeds preserve fixed coordinates and do not delete single atom motion", "[neb][seeds]") {
    auto path=seed_band(false);for(auto &a:path)a.sel_dynpos_={Eigen::RowVector3d(1,0,1),Eigen::RowVector3d::Zero()};
    NEBDriver d;d.sel_dyn=true;d.prepare_complete_path(path,path[0].cell_);NEBSeedOptions o;o.atoms={0};
    auto r=NEBPathSeeds::generate(d,o);REQUIRE(r.status=="generated");REQUIRE(r.selected_atoms==std::vector<int>{0});
    for(int i=0;i<7;++i){REQUIRE(r.images[i].pos_[1]==path[i].pos_[1]);REQUIRE(r.images[i].pos_[0][1]==path[i].pos_[0][1]);}
    o.atoms={1};REQUIRE(NEBPathSeeds::generate(d,o).status=="skipped");
    o.atoms={};o.center=0;o.radius=.5;r=NEBPathSeeds::generate(d,o);REQUIRE(r.status=="generated");REQUIRE(r.selected_atoms==std::vector<int>{0});
}
TEST_CASE("Cell seed amplitude and source coordinate chart are preserved", "[neb][seeds][vcneb]") {
    auto path=seed_band();for(int i=0;i<7;++i){Eigen::Matrix3d cell=(1+.002*i)*path[i].cell_;path[i].set_cell(cell);}
    NEBDriver d;d.vcneb=true;d.apply_scal=true;d.ciival={80,120,90,40,50,60};d.prepare_complete_path(path,path[0].cell_);
    NEBSeedOptions o;o.amplitude=0;auto r=NEBPathSeeds::generate(d,o);
    REQUIRE(r.status=="generated");REQUIRE(r.cell_deformation<=.00100000001);REQUIRE(r.cell_deformation>.00099);
    REQUIRE(r.images.front().cell_==path.front().cell_);REQUIRE(r.images.back().cell_==path.back().cell_);
    REQUIRE(r.atomic_displacement<1e-12);
    NEBPathSymmetry projector(d,"");NEBDriver after;after.vcneb=true;after.apply_scal=true;after.ciival=d.ciival;after.prepare_complete_path(r.images,d.refcell);
    for(int i=1;i<6;++i)REQUIRE(projector.project((after.x.row(i)-d.x.row(i)).cwiseProduct(d.scalval.transpose())).norm()<1e-9);
}

TEST_CASE("Complete VC paths accept comma-separated metric weights", "[neb][seeds][vcneb]") {
    NEBDriver d;d.vcneb=true;d.apply_scal=true;d.ciistr="80,120,90,40,50,60";
    auto path=seed_band();d.prepare_complete_path(path,path[0].cell_);
    REQUIRE(d.scalval[0]==Catch::Approx(std::sqrt(80.)));
    REQUIRE(d.scalval[1]==Catch::Approx(std::sqrt(120.)));
}
