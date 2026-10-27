#include <catch2/catch_all.hpp>
#include "mindriver/relax_preconditioner.h"
#include "potentials/potential.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace RelaxPreconditioning;
using Catch::Approx;
namespace {
struct MatrixFile {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("imrmlp-full-preconditioner-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    void write(const std::string &text) { std::ofstream out(path);out<<text; }
    ~MatrixFile() { std::error_code ec;std::filesystem::remove(path,ec); }
};
Atoms chain() {
    Atoms a;a.cell_=Eigen::Matrix3d::Identity()*10;a.pbc_={false,false,false};
    a.pos_={{0,0,0},{1,0,0},{2.5,0,0}};a.chemmap_={0,0,0};return a;
}
class NativeCurvature : public Potentials::AnyPotential {
public:
    bool dense=true,hvp=true,bad=false;int dense_calls=0,hvp_calls=0,efs_calls=0;
    NativeCurvature() { cutoff=4; }
    Eigen::MatrixXd value(const Atoms &a) const {
        Eigen::MatrixXd h=Eigen::MatrixXd::Identity(3*a.size(),3*a.size())*(2+a.pos_[0][0]);
        h(0,1)=h(1,0)=.3;return h;
    }
    void getEFS(Atoms &,AnyNeighborhoods &) override { ++efs_calls;throw std::runtime_error("no force FD"); }
    bool supports_analytical_hessian() const override { return dense; }
    bool supports_analytical_hessian_vector_product() const override { return hvp; }
    void getEFSHessian(Atoms &a,AnyNeighborhoods &,Potentials::HessianMatrix &blocks) override {
        ++dense_calls;if (bad) throw std::runtime_error("native curvature failure");
        const Eigen::MatrixXd h=value(a);blocks.resize(a.size());
        for (int i=0;i<a.size();++i) for (int j=0;j<a.size();++j) blocks(i,j)=h.block<3,3>(3*i,3*j);
        a.energy_=-200;a.pos_[0][0]=100; // must only mutate helper's owned copy
    }
    void applyHessianVectorProduct(Atoms &a,AnyNeighborhoods &,const Eigen::VectorXd &direction,Eigen::VectorXd &product) override {
        ++hvp_calls;if (bad) {product=Eigen::VectorXd::Zero(1);return;}product=value(a)*direction;
    }
};
}

TEST_CASE("Full Hessian file is separate row-major strict input", "[relax_preconditioner][full_metric]") {
    MatrixFile f;f.write("3\n4 1 2\n1 5 3\n2 3 6\n");
    Eigen::Matrix3d expected;expected<<4,1,2,1,5,3,2,3,6;
    REQUIRE(read_full_hessian(f.path.string(),3).isApprox(expected));
    REQUIRE_THROWS(read_full_hessian(f.path.string(),2));
    for (const auto &bad:{"2 1 0 0", "2 1 2 0 1", "2 1 0 0 nan", "2 1 0 0 1 extra", "-1"}) {
        f.write(bad);REQUIRE_THROWS(read_full_hessian(f.path.string(),2));
    }
    REQUIRE_THROWS(read_full_hessian(f.path.string(),4097));
}
TEST_CASE("Full metric retains cross terms through LF LC and selective pullbacks", "[relax_preconditioner][full_metric]") {
    Eigen::Matrix3d h;h<<5,2,-1,2,4,.5,-1,.5,3;
    // LF: physical x=A^-1 u; LC: u=Qv. These fixed linear maps admit exact J^T H J.
    Eigen::Matrix3d a;a<<2,.4,0,0,1.5,.2,0,0,3;
    Eigen::Matrix<double,3,2> q;q<<1,0,0,1,.5,-.25;
    const Eigen::MatrixXd j=a.inverse()*q, transformed=pullback(pullback(h,a.inverse()),q);
    REQUIRE(transformed.isApprox(j.transpose()*h*j,1e-12));
    Eigen::Vector2d u;u<<.7,-.2;const double step=1e-5;
    const auto energy=[&](Eigen::Vector2d v) {Eigen::Vector3d x=j*v;return .5*x.dot(h*x);};
    const Eigen::VectorXd gradient=transformed*u;
    for (int i=0;i<2;++i) {
        Eigen::Vector2d p=u,m=u;p[i]+=step;m[i]-=step;
        REQUIRE((energy(p)-energy(m))/(2*step)==Approx(gradient[i]).margin(1e-9));
        REQUIRE(((transformed*p-transformed*m)/(2*step)).isApprox(transformed.col(i),1e-9));
    }
    const Eigen::MatrixXd masked=select_coordinates(h,{2,0});
    REQUIRE(masked(0,1)==-1);REQUIRE(masked(0,0)==3);REQUIRE(masked(1,1)==5);
    REQUIRE(select_coordinates(h,{}).size()==0);
    REQUIRE_THROWS(select_coordinates(h,{0,0}));REQUIRE_THROWS(select_coordinates(h,{3}));
    REQUIRE_THROWS(pullback(h,Eigen::MatrixXd::Identity(2,2)));
}
TEST_CASE("Spectral absolute metric whitens without changing physical gradients", "[relax_preconditioner][full_metric]") {
    Eigen::Matrix3d u;u=Eigen::AngleAxisd(.43,Eigen::Vector3d(1,2,3).normalized());
    const Eigen::Matrix3d h=u*Eigen::Vector3d(-4,0,9).asDiagonal()*u.transpose();
    DenseMetric metric(h,.25);const Eigen::Matrix3d expected=u*Eigen::Vector3d(4,.25,9).asDiagonal()*u.transpose();
    REQUIRE(metric.matrix().isApprox(expected,1e-12));
    REQUIRE((metric.root()*metric.root()).isApprox(expected,1e-12));
    REQUIRE((metric.root()*metric.inverse_root()).isApprox(Eigen::Matrix3d::Identity(),1e-12));
    Eigen::Vector3d g(2,-1,.4);REQUIRE(metric.solve(g).isApprox(expected.ldlt().solve(g),1e-12));
    metric.reset(metric.matrix(),.25); // input may alias stored matrix
    REQUIRE(metric.matrix().isApprox(expected,1e-12));
    const Eigen::MatrixXd before=metric.matrix();REQUIRE_THROWS(metric.reset(h,0));REQUIRE(metric.matrix().isApprox(before));
    Eigen::Vector3d origin(.2,.5,-.1),physical(1,.2,.3);
    WhiteningTransform white(h,origin,.25);Eigen::VectorXd z=white.reduce(physical);
    REQUIRE(white.lift(z).isApprox(physical,1e-12));
    const Eigen::VectorXd gz=white.gradient(h*physical);const double step=1e-6;
    for (int i=0;i<3;++i) {
        Eigen::VectorXd p=z,m=z;p[i]+=step;m[i]-=step;
        const Eigen::VectorXd xp=white.lift(p),xm=white.lift(m);
        REQUIRE((.5*xp.dot(h*xp)-.5*xm.dot(h*xm))/(2*step)==Approx(gz[i]).margin(1e-8));
    }
    REQUIRE(g.isApprox(Eigen::Vector3d(2,-1,.4)));
}
TEST_CASE("Sparse Exp metric matches independent graph and solves active coordinates", "[relax_preconditioner][sparse_metric]") {
    Atoms a=chain();SparseOptions options;options.shift=.2;
    SparseMetric sparse(a,3,options);REQUIRE(sparse.nearest_scale()==Approx(1));REQUIRE(sparse.cutoff()==Approx(2));
    const double w=std::exp(-1.5);Eigen::Matrix3d graph;
    graph<<1.2,-1,0,-1,1.2+w,-w,0,-w,.2+w;
    Eigen::MatrixXd expected=Eigen::MatrixXd::Zero(9,9);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) expected.block<3,3>(3*i,3*j)=graph(i,j)*Eigen::Matrix3d::Identity();
    REQUIRE(Eigen::MatrixXd(sparse.matrix()).isApprox(expected,1e-12));
    Eigen::VectorXd g(9);g<<1,2,3,4,-2,.3,2,.7,-4;
    REQUIRE(sparse.solve(g).isApprox(expected.ldlt().solve(g),1e-9));
    REQUIRE((sparse.matrix()*Eigen::VectorXd::Ones(9)).isApprox(Eigen::VectorXd::Constant(9,.2),1e-12));
    REQUIRE(sparse.matrix().nonZeros()<=21);
    SparseMetric masked(a,3,options,{7,0,3});const Eigen::MatrixXd block=select_coordinates(expected,{7,0,3});
    REQUIRE(Eigen::MatrixXd(masked.matrix()).isApprox(block));
    REQUIRE(masked.solve(Eigen::Vector3d(1,-2,3)).isApprox(block.ldlt().solve(Eigen::Vector3d(1,-2,3)),1e-9));
    REQUIRE(a.pos_[2].isApprox(Eigen::RowVector3d(2.5,0,0)));REQUIRE(a.chemmap_==std::vector<int>{0,0,0});
    REQUIRE_THROWS(sparse.reset(a,.2,options));REQUIRE(Eigen::MatrixXd(sparse.matrix()).isApprox(expected));
    REQUIRE_THROWS(SparseMetric(a,3,options,{0,0}));
    options.max_iterations=1;options.relative_tolerance=1e-14;SparseMetric short_solve(a,3,options);
    REQUIRE_THROWS(short_solve.solve(g));
}
TEST_CASE("Sparse periodic images accumulate edges and self images cancel", "[relax_preconditioner][sparse_metric]") {
    Atoms a;a.cell_=Eigen::Matrix3d::Zero();a.cell_(0,0)=2;a.pbc_={true,false,false};
    a.pos_={{0,0,0},{1,0,0}};a.chemmap_={0,0};SparseOptions options;options.shift=.1;
    SparseMetric metric(a,1.1,options);const Eigen::MatrixXd dense(metric.matrix());
    // Two images of the other atom at distance1, each weight1.
    REQUIRE(dense(0,0)==Approx(2.1));REQUIRE(dense(0,3)==Approx(-2));
    a.pos_.resize(1);a.chemmap_.resize(1);SparseMetric one(a,2.1,options);
    REQUIRE(Eigen::MatrixXd(one.matrix()).isApprox(Eigen::Matrix3d::Identity()*.1));
    a.pbc_={false,false,false};SparseMetric isolated(a,1,options);
    REQUIRE(Eigen::MatrixXd(isolated.matrix()).isApprox(Eigen::Matrix3d::Identity()*.1));
}
TEST_CASE("Periodic Gram scaling handles thin lower dimensional cells without full inverse", "[relax_preconditioner][aul_auto]") {
    Eigen::Matrix3d cell=Eigen::Matrix3d::Zero();cell.row(0)<<1e-6,0,0;cell.row(2)<<1,2,0;
    const auto gram=periodic_gram(cell,{true,false,true});
    REQUIRE(gram.axes==std::vector<int>{0,2});REQUIRE((gram.rows*gram.dual).isApprox(Eigen::Matrix2d::Identity(),1e-10));
    REQUIRE(gram.heights[0]==Approx(2e-6/std::sqrt(5.)).epsilon(1e-10));REQUIRE(gram.heights[1]==Approx(2));
    REQUIRE(gram.condition>1e12);
    const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.61,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    const auto rotated=periodic_gram(cell*rotation,{true,false,true});
    REQUIRE(rotated.gram.isApprox(gram.gram,1e-12));REQUIRE(rotated.heights.isApprox(gram.heights,1e-12));
    REQUIRE(periodic_gram(cell,{false,false,false}).dual.cols()==0);
    REQUIRE(periodic_gram(cell,{true,false,false}).heights[0]==Approx(1e-6));
    cell.row(2)=2*cell.row(0);REQUIRE_THROWS(periodic_gram(cell,{true,false,true}));
    REQUIRE_THROWS(periodic_gram(cell,{false,true,false}));
}
TEST_CASE("AUL auto scales constraint compliance and diagnoses impossible initial residual", "[relax_preconditioner][aul_auto]") {
    Eigen::Matrix2d j;j<<2,0,0,3;Eigen::Vector2d residual(4,6);
    auto plain=estimate_aul(j,residual);REQUIRE(plain.weights.isApprox(Eigen::Vector2d(.25,1./9)));
    REQUIRE(plain.correction_norm.isApprox(Eigen::Vector2d(2,2)));
    const auto solve=[](const Eigen::VectorXd &g)->Eigen::VectorXd {return g.cwiseQuotient(Eigen::Vector2d(4,9));};
    auto metric=estimate_aul(j,residual,solve);REQUIRE(metric.weights.isApprox(Eigen::Vector2d::Ones()));
    REQUIRE(metric.correction_norm.isApprox(residual));
    auto scaled=estimate_aul(j*10,residual*10);REQUIRE(scaled.correction_norm.isApprox(plain.correction_norm));
    REQUIRE((scaled.weights*100).isApprox(plain.weights));
    j.row(0).setZero();residual[0]=0;REQUIRE(estimate_aul(j,residual).weights[0]==0);
    residual[0]=1;REQUIRE_THROWS(estimate_aul(j,residual));
    j.setIdentity();REQUIRE_THROWS(estimate_aul(j,residual,[](const Eigen::VectorXd &g)->Eigen::VectorXd{return -g;}));
}
TEST_CASE("Native FC refresh prefers dense or exact HVP and never falls back to force FD", "[relax_preconditioner][online_fc]") {
    Atoms a=chain();a.energy_=12;a.forces_.assign(a.size(),Eigen::Vector3d(1,2,3));
    NativeCurvature potential;const Eigen::MatrixXd expected=potential.value(a);
    REQUIRE(native_cartesian_hessian(potential,a).isApprox(expected));
    REQUIRE(potential.dense_calls==1);REQUIRE(potential.hvp_calls==0);REQUIRE(potential.efs_calls==0);
    REQUIRE(a.energy_==12);REQUIRE(a.pos_[0].isZero());REQUIRE(a.forces_[0].isApprox(Eigen::Vector3d(1,2,3)));
    potential.dense=false;REQUIRE(native_cartesian_hessian(potential,a).isApprox(expected));
    REQUIRE(potential.hvp_calls==9);REQUIRE(potential.efs_calls==0);
    potential.bad=true;REQUIRE_THROWS(native_cartesian_hessian(potential,a));
    potential.bad=false;potential.hvp=false;REQUIRE_THROWS(native_cartesian_hessian(potential,a));
    REQUIRE(potential.efs_calls==0);REQUIRE_THROWS(native_cartesian_hessian(potential,a,8));
}
TEST_CASE("Accepted FC update recomputes true curvature only once per accepted interval", "[relax_preconditioner][online_fc]") {
    NativeCurvature potential;Atoms a=chain();AcceptedFCUpdate update(2);
    REQUIRE_FALSE(update.compute(potential,a,0));REQUIRE_FALSE(update.compute(potential,a,1));REQUIRE(potential.dense_calls==0);
    const auto initial=update.compute(potential,a,2);REQUIRE(initial);REQUIRE(potential.dense_calls==1);
    REQUIRE_FALSE(update.compute(potential,a,2));a.pos_[0][0]=.5;potential.bad=true;
    REQUIRE_THROWS(update.compute(potential,a,4));REQUIRE(update.due(4));
    potential.bad=false;const auto refreshed=update.compute(potential,a,4);REQUIRE(refreshed);
    REQUIRE(refreshed->isApprox(potential.value(a)));REQUIRE_FALSE(refreshed->isApprox(*initial));REQUIRE_FALSE(update.due(4));
    REQUIRE(potential.efs_calls==0);update.configure(0);REQUIRE_FALSE(update.compute(potential,a,6));
    REQUIRE_THROWS(update.configure(-1));
}
