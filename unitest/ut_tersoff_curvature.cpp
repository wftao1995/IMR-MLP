#include <catch2/catch_all.hpp>
#include "potentials/potential_tersoff.h"
#include "unitest/pair_cell_fixture.h"
#include <limits>
#include <map>
#include <sstream>

using namespace Potentials;
namespace {
void load(Tersoff& model, bool brenner) {
    std::ostringstream out;
    out << "mode=" << (brenner ? "brenner" : "tersoff") << '\n';
    const char* names[] = {"H","He"};
    for (int i = 0; i < 2; ++i)
        for (int j = i; j < 2; ++j)
            out << names[i] << ' ' << names[j]
                << " A=" << 2.3+.3*i+.2*j << " B=" << 1.1+.2*i+.1*j
                << " lambda1=1.4 lambda2=.8 lambda3=.18 m=3 n=1.4 beta=.7"
                << " D0=" << 1.2+.2*i+.1*j << " S=1.8 r0=1.4 gamma=" << .7+.2*i+.1*j
                << " R=" << 3.+.05*(i+j) << " D=.55 c=.7 d=.8 h=-.2\n";
    std::istringstream input(out.str());
    model.Load(input);
    model.train_init();
}
Eigen::VectorXd force_fd(AnyPotential& model, const Atoms& atom,
    const Eigen::VectorXd& direction, double step, double cutoff) {
    auto plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step*direction.segment<3>(3*i).transpose();
        minus.pos_[i] -= step*direction.segment<3>(3*i).transpose();
    }
    Neighborhoods np(plus,cutoff), nm(minus,cutoff);
    model.getEFS(plus,np);
    model.getEFS(minus,nm);
    Eigen::VectorXd result(direction.size());
    for (int i = 0; i < atom.size(); ++i)
        result.segment<3>(3*i) = -(plus.forces_[i]-minus.forces_[i])/(2*step);
    return result;
}
void same_efs(const Atoms& actual, const Atoms& expected, double factor = 1.) {
    REQUIRE(std::abs(actual.energy_-factor*expected.energy_) < 1e-10);
    REQUIRE((actual.stress_-factor*expected.stress_).norm() < 1e-10);
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE((actual.forces_[i]-factor*expected.forces_[i]).norm() < 1e-10);
}
}

TEST_CASE("Tersoff and Brenner HVP dense and cell derivatives retain bond order curvature",
          "[tersoff][brenner][hvp][hessian][cell]") {
    const bool brenner = GENERATE(false,true);
    const bool partial = GENERATE(false,true);
    INFO("Brenner=" << brenner << " partial PBC=" << partial);
    Tersoff model; load(model,brenner); AnyPotential& api = model;
    REQUIRE(api.supports_analytical_hessian_vector_product());
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_cell_hessian());
    REQUIRE(api.supports_site_hvp());
    REQUIRE(api.supports_site_hessian());
    auto atom = PairCellTest::geometry(partial), reference = atom;
    Neighborhoods nbhs(atom,api.cutoff);
    bool repeated = false;
    for (int i = 0; i < nbhs.size(); ++i) {
        std::map<int,int> counts;
        for (int source : nbhs[i].indices_) repeated |= ++counts[source] > 1;
    }
    REQUIRE(repeated);
    std::vector<double> coefficients(api.get_NL_coeffs_count());
    api.nl_get_coeffs(coefficients.data());
    api.getEFS(reference,nbhs);
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    api.getEFSHessianCell(atom,nbhs,h,cell,cross);
    const auto dense = HessianUtils::to_dense(h);
    const auto native = PairCellTest::native_matrix(h,cell,cross);
    same_efs(atom,reference);
    REQUIRE((dense-dense.transpose()).norm() < 1e-10);
    REQUIRE((cell-cell.transpose()).norm() < 1e-10);
    REQUIRE(cell.norm() > 1e-6);
    REQUIRE(cross.norm() > 1e-6);
    for (int probe = 1; probe <= 2; ++probe) {
        Eigen::VectorXd direction(3*atom.size()), result;
        for (int i = 0; i < direction.size(); ++i) direction[i] = std::sin(.71*(i+1)*probe);
        const auto before = atom;
        api.applyHessianVectorProduct(atom,nbhs,direction,result);
        REQUIRE(atom.pos_ == before.pos_);
        same_efs(atom,before);
        REQUIRE((dense*direction-result).norm() < 1e-10);
        for (double step : {1e-5,5e-6}) {
            const auto expected = force_fd(api,atom,direction,step,api.cutoff);
            REQUIRE((result-expected).norm() < 2e-7*std::max(1.,expected.norm()));
        }
        auto alias = direction;
        api.applyHessianVectorProduct(atom,nbhs,alias,alias);
        REQUIRE((alias-result).norm() < 1e-12);
        // Relabeling atoms and their probes must merely permute the result.
        auto permuted = atom;
        std::swap(permuted.pos_[0],permuted.pos_[2]);
        std::swap(permuted.chemmap_[0],permuted.chemmap_[2]);
        Eigen::VectorXd permuted_direction = direction, permuted_result;
        permuted_direction.segment<3>(0) = direction.segment<3>(6);
        permuted_direction.segment<3>(6) = direction.segment<3>(0);
        api.applyHessianVectorProduct(permuted,permuted_direction,permuted_result);
        REQUIRE((permuted_result.segment<3>(0)-result.segment<3>(6)).norm() < 1e-10);
        REQUIRE((permuted_result.segment<3>(3)-result.segment<3>(3)).norm() < 1e-10);
        REQUIRE((permuted_result.segment<3>(6)-result.segment<3>(0)).norm() < 1e-10);
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(3*atom.size()), product;
        for (int i = 0; i < atom.size(); ++i) translation[3*i+axis] = 1.;
        api.applyHessianVectorProduct(atom,nbhs,translation,product);
        REQUIRE(product.isZero(1e-12));
        REQUIRE((dense*translation).norm() < 1e-10);
    }
    for (double step : {.001,.0005}) {
        const auto expected = PairCellTest::energy_fd(api,atom,step,true);
        REQUIRE((native-expected).norm() < 150*step*step*std::max(1.,expected.norm())+2e-6);
    }
    for (int route = 0; route < 8; ++route) {
        auto work = atom;
        if (route == 0) api.getEFSHessian(work,h);
        if (route == 1) api.getEFSHessian(work,nbhs,h);
        if (route == 2) api.getEFSHessianOMP(work,nbhs,h);
        if (route == 3) api.getEFSHessianSymm(work,h);
        if (route == 4) api.getEFSHessianSymmOMP(work,h);
        if (route == 5) api.getEFSHessianCell(work,h,cell,cross);
        if (route == 6) api.getEFSHessianCellOMP(work,nbhs,h,cell,cross);
        if (route == 7) api.getEFSHessianCellSymmOMP(work,h,cell,cross);
        REQUIRE((HessianUtils::to_dense(h)-dense).norm() < 1e-10);
        same_efs(work,reference);
    }
    api.add_efs = true;
    api.getEFSHessianCellSymm(atom,h,cell,cross);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-native).norm() < 1e-10);
    same_efs(atom,reference,2.);
    std::vector<double> after(coefficients.size());
    api.nl_get_coeffs(after.data());
    REQUIRE(after == coefficients);
}

TEST_CASE("Bond-order second derivative couples different zeta neighbors",
          "[tersoff][brenner][hessian][zeta]") {
    const bool brenner = GENERATE(false,true);
    Tersoff model;
    std::ostringstream out;
    out << "mode=" << (brenner ? "brenner" : "tersoff") << '\n'
        << "H H A=0 B=2 lambda1=0 lambda2=0 lambda3=0 n=1.4 beta=" << (brenner ? 0. : .7)
        << " D0=1 S=2 r0=1 gamma=1 R=3 D=.55 c=0 d=1 h=0\n";
    std::istringstream input(out.str()); model.Load(input);
    Atoms atom;
    atom.cell_ = 10.*Eigen::Matrix3d::Identity();
    atom.pbc_ = {false,false,false};
    atom.pos_ = {{0,0,0},{3,0,0},{0,3,0},{0,0,3}};
    atom.chemmap_ = {0,0,0,0};
    Neighborhoods nbhs(atom,model.cutoff);
    REQUIRE(nbhs[0].nbhcount_ == 3);
    const double f = .5, fp = -M_PI/(4*.55), zeta = 2*f;
    double first,second;
    if (brenner) {
        first = -.5*std::pow(1.+zeta,-1.5);
        second = .75*std::pow(1.+zeta,-2.5);
    } else {
        const double n = 1.4, q = -1./(2*n), bn = std::pow(.7,n);
        const double base = 1.+bn*std::pow(zeta,n), b = std::pow(base,q);
        const double dz = bn*n*std::pow(zeta,n-1), dzz = bn*n*(n-1)*std::pow(zeta,n-2);
        first = b*q*dz/base;
        second = b*(q*(q-1)*dz*dz/(base*base)+q*dzz/base);
    }
    const double expected = -.5*2.*fp*fp*(2*first+f*second);
    const double missing_second = -.5*2.*fp*fp*2*first;
    REQUIRE(std::abs(expected-missing_second) > 1e-3);
    HessianMatrix h; double energy;
    SiteCoordinateMatrix gradient;
    model.evaluate_site_hessian(nbhs[0],energy,gradient,h);
    int x = -1,y = -1;
    for (int j = 0; j < nbhs[0].nbhcount_; ++j) {
        if (nbhs[0].indices_[j] == 1) x = j;
        if (nbhs[0].indices_[j] == 2) y = j;
    }
    REQUIRE(x >= 0); REQUIRE(y >= 0);
    REQUIRE(h(x,0,y,1) == Catch::Approx(expected).margin(1e-11));
    SiteCoordinateMatrix direction = SiteCoordinateMatrix::Zero(3,3), product,directional_gradient;
    direction(y,1) = 1.;
    model.apply_site_hvp(nbhs[0],direction,energy,directional_gradient,product);
    REQUIRE(product(x,0) == Catch::Approx(expected).margin(1e-11));
}

TEST_CASE("Tersoff and Brenner periodic self images retain cell curvature and HVP contracts",
          "[tersoff][brenner][cell][self-image]") {
    const bool brenner = GENERATE(false,true);
    Tersoff model; load(model,brenner); AnyPotential& api = model;
    Atoms atom;
    atom.cell_ = 2.8*Eigen::Matrix3d::Identity();
    atom.pos_ = {{.2,.3,.4}};
    atom.chemmap_ = {0};
    Neighborhoods nbhs(atom,api.cutoff);
    REQUIRE(nbhs[0].nbhcount_ > 1);
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    api.getEFSHessianCellSymm(atom,h,cell,cross);
    REQUIRE(h(0,0).isZero(1e-11));
    REQUIRE(cross.isZero(1e-11));
    REQUIRE(cell.norm() > 1e-6);
    for (double step : {.001,.0005}) {
        const auto expected = PairCellTest::energy_fd(api,atom,step,true);
        REQUIRE((PairCellTest::native_matrix(h,cell,cross)-expected).norm() < 150*step*step*std::max(1.,expected.norm())+2e-6);
    }
    Eigen::VectorXd direction = Eigen::VectorXd::Ones(3), product;
    api.applyHessianVectorProduct(atom,nbhs,direction,product);
    REQUIRE(product.isZero(1e-12));
    auto invalid = Eigen::VectorXd::Zero(4).eval();
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom,nbhs,invalid,product));
    invalid = direction;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom,nbhs,invalid,product));
    atom.cell_.setZero();
    REQUIRE_THROWS(api.getEFSHessianCellOMP(atom,nbhs,h,cell,cross));
}
