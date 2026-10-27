#include <catch2/catch_all.hpp>
#include "mindriver/neb_preconditioner.h"

TEST_CASE("NEB retained model directions select resolved soft modes", "[neb][stable][preconditioner]") {
    Eigen::MatrixXd directions=Eigen::MatrixXd::Identity(4,4);
    Eigen::MatrixXd model=Eigen::Vector4d(1e-14,.01,.1,10).asDiagonal();
    const auto retained=NEBNumerics::soft_directions(directions,model,2);
    REQUIRE(retained.cols()==2);
    CHECK((retained.transpose()*retained).isApprox(Eigen::Matrix2d::Identity(),1e-12));
    CHECK(std::abs(retained(1,0))==Catch::Approx(1));
    CHECK(std::abs(retained(2,1))==Catch::Approx(1));
    CHECK(retained.row(0).norm()==0);
    CHECK(retained.row(3).norm()==0);
    model.setZero();
    CHECK(NEBNumerics::soft_directions(directions,model,2).cols()==0);
}

TEST_CASE("NEB atomic graph preserves translations and atom permutation", "[neb][stable][preconditioner]") {
    Atoms atoms;atoms.cell_<<6,0,0,1.5,5,0,0,0,6;atoms.pbc_={true,true,true};
    atoms.chemmap_={0,0,0,0};
    atoms.pos_={{.1,.3,1},{5.9,.3,1},{2.5,2,1},{4,4,3}};
    const auto original=atoms.pos_;
    NEBNumerics::AtomicGraph graph;
    REQUIRE(graph.build(atoms,4));
    CHECK(atoms.pos_==original);
    Eigen::MatrixXd rhs(4,3);rhs<<1,2,3,-1,3,1,2,-1,4,-2,-4,1;
    const Eigen::MatrixXd solution=graph.solve(rhs);
    REQUIRE(solution.allFinite());
    for(int j=0;j<3;++j)CHECK(rhs.col(j).dot(solution.col(j))>0);
    CHECK(graph.solve(Eigen::MatrixXd::Ones(4,3)).isApprox(10*Eigen::MatrixXd::Ones(4,3),1e-11));
    Atoms reordered=atoms;Eigen::MatrixXd permuted(4,3);
    const std::array<int,4> order{2,0,3,1};
    for(int i=0;i<4;++i) {
        reordered.pos_[i]=atoms.pos_[order[i]]+2*atoms.cell_.row(0)-atoms.cell_.row(1);
        permuted.row(i)=rhs.row(order[i]);
    }
    NEBNumerics::AtomicGraph other;
    REQUIRE(other.build(reordered,4));
    const Eigen::MatrixXd actual=other.solve(permuted);
    for(int i=0;i<4;++i)CHECK(actual.row(i).isApprox(solution.row(order[i]),1e-10));
}

TEST_CASE("NEB atomic graph safely declines empty or invalid neighborhoods", "[neb][stable][preconditioner]") {
    Atoms atoms;atoms.cell_=10*Eigen::Matrix3d::Identity();atoms.pbc_={false,false,false};
    atoms.chemmap_={0};atoms.pos_={{0,0,0}};
    NEBNumerics::AtomicGraph graph;
    CHECK_FALSE(graph.build(atoms,4));
    atoms.chemmap_.push_back(0);atoms.pos_.push_back({1,0,0});
    CHECK_FALSE(graph.build(atoms,-1));
    CHECK_FALSE(graph.valid());
}
