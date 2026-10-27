#include <catch2/catch_all.hpp>
#include "symmetry/symmetry.h"

namespace {
Atoms slab(int multiplicity, int first=0, int second=1, bool singular=false) {
    Atoms result; result.pbc_={false,false,false};result.pbc_[first]=result.pbc_[second]=true;
    const Eigen::Matrix3d rotate=Eigen::AngleAxisd(.31,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    const Eigen::RowVector3d a=Eigen::RowVector3d(2,0,0)*rotate;
    const Eigen::RowVector3d b=Eigen::RowVector3d(.4,2.5,0)*rotate;
    const Eigen::RowVector3d normal=a.cross(b).normalized();
    result.cell_.row(first)=multiplicity==2?(a+b).eval():(2*a+b).eval();
    result.cell_.row(second)=multiplicity==2?(-a+b).eval():(2*b).eval();
    result.cell_.row(3-first-second)=singular?Eigen::RowVector3d::Zero().eval():(15*normal+.2*a+.3*b).eval();
    const std::vector<Eigen::RowVector3d> shifts=multiplicity==2
        ?std::vector<Eigen::RowVector3d>{Eigen::RowVector3d::Zero(),a}
        :std::vector<Eigen::RowVector3d>{Eigen::RowVector3d::Zero(),a,b,a+b};
    for(const auto &shift:shifts) {
        result.pos_.push_back(.2*a+.3*b+.4*normal+shift);result.chemmap_.push_back(0);
        result.pos_.push_back(.6*a+.7*b+1.2*normal+shift);result.chemmap_.push_back(1);
    }
    result.id=17;result.features_["label"]="slab";result.features_["config_type"]="layered";
    result.have_energy=result.have_fenergy=result.have_stress=true;
    result.energy_=16.;result.fenergy_=12.;result.stress_=Eigen::Matrix3d::Identity()*8;
    result.global_attribs_d_["total_charge"]=4.;
    result.forces_.assign(result.size(),Eigen::Vector3d::Ones());
    result.have_hessian=true;result.hessian_=Eigen::MatrixXd::Identity(3*result.size(),3*result.size());
    return result;
}
}

TEST_CASE("2D primitive search finds the whole non-diagonal plane translation group", "[symmetry][primitive2d]") {
    const int multiplicity=GENERATE(2,4);
    const int first=GENERATE(0,1,2),second=(first+1)%3;
    const bool fractional=GENERATE(false,true);
    auto atoms=slab(multiplicity,first,second);
    if(fractional) atoms.to_fractional();
    Atoms original=atoms;if(!original.cartesian) original.to_cartesian();
    const auto open=atoms.cell_.row(3-first-second).eval();
    const Eigen::RowVector3d normal=atoms.cell_.row(first).cross(atoms.cell_.row(second)).normalized();
    const double area=atoms.cell_.row(first).cross(atoms.cell_.row(second)).norm();
    REQUIRE(Symmetry3D::find_primitive_2d(atoms,first,second)==multiplicity);
    REQUIRE(atoms.size()==2);REQUIRE(atoms.cartesian==!fractional);
    REQUIRE(atoms.pbc_==original.pbc_);REQUIRE(atoms.cell_.row(3-first-second)==open);
    REQUIRE(atoms.cell_.row(first).cross(atoms.cell_.row(second)).norm()==Catch::Approx(area/multiplicity));
    Atoms reduced=atoms;if(!reduced.cartesian) reduced.to_cartesian();
    Eigen::Matrix<double,2,3> basis;basis.row(0)=reduced.cell_.row(first);basis.row(1)=reduced.cell_.row(second);
    const Eigen::Matrix<double,2,3> dual=(basis*basis.transpose()).inverse()*basis;
    Eigen::Matrix2d integer;
    integer.row(0)=original.cell_.row(first)*dual.transpose();
    integer.row(1)=original.cell_.row(second)*dual.transpose();
    REQUIRE(integer.isApprox(integer.array().round().matrix(),1e-12));
    REQUIRE(integer.determinant()==Catch::Approx(multiplicity));
    for(int i=0;i<original.size();++i) {
        bool found=false;
        for(int j=0;j<reduced.size();++j) if(original.chemmap_[i]==reduced.chemmap_[j]) {
            const auto delta=(original.pos_[i]-reduced.pos_[j]).eval();
            const Eigen::RowVector2d fraction=delta*dual.transpose();
            if(std::abs(delta.dot(normal))<1e-12 && (fraction-fraction.array().round().matrix()).norm()<1e-12) found=true;
        }
        REQUIRE(found);
    }
    REQUIRE(atoms.id==17);REQUIRE(atoms.features_.at("label")=="slab");
    REQUIRE(atoms.energy_==Catch::Approx(16./multiplicity));
    REQUIRE(atoms.fenergy_==Catch::Approx(12./multiplicity));
    REQUIRE(atoms.stress_(0,0)==Catch::Approx(8./multiplicity));
    REQUIRE(atoms.global_attribs_d_.at("total_charge")==Catch::Approx(4./multiplicity));
    REQUIRE(atoms.forces_.empty());REQUIRE_FALSE(atoms.have_hessian);
    REQUIRE(Symmetry3D::find_primitive_2d(atoms,first,second)==1);
}

TEST_CASE("2D primitive reduction supports Cartesian slabs with a missing open row", "[symmetry][primitive2d]") {
    auto atoms=slab(2,0,1,true);
    REQUIRE(Symmetry3D::find_primitive_2d(atoms,0,1)==2);
    REQUIRE(atoms.size()==2);REQUIRE(atoms.cell_.row(2).isZero());REQUIRE(atoms.cartesian);
}

TEST_CASE("A translation that fails in another layer or species does not reduce the slab", "[symmetry][primitive2d]") {
    auto atoms=slab(2);const bool change_species=GENERATE(false,true);
    if(change_species) atoms.chemmap_.back()=2;
    else atoms.pos_.back()+=.1*atoms.cell_.row(0).cross(atoms.cell_.row(1)).normalized();
    const Atoms original=atoms;
    REQUIRE(Symmetry3D::find_primitive_2d(atoms,0,1)==1);
    REQUIRE(atoms.pos_==original.pos_);REQUIRE(atoms.cell_==original.cell_);
    REQUIRE(atoms.hessian_==original.hessian_);REQUIRE(atoms.energy_==16.);
}
