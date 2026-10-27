#include <catch2/catch_all.hpp>
#include "potentials/potential_mtpnn1.h"
#include "unitest/many_body_mtp_fixture.h"
#include "unitest/pair_cell_fixture.h"
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>

using namespace Potentials;
namespace {
struct NeuralFixture {
    std::filesystem::path directory;
    NeuralFixture() {
        char pattern[] = "/tmp/imrmlp_legacy_nn_curvature_XXXXXX";
        const char* made = mkdtemp(pattern);
        if (!made) throw std::runtime_error("mkdtemp failed");
        directory = made;
        PotMTP classic;
        load_many_body_mtp_fixture(classic,2);
        classic.Save((directory/"model.mtp").string());
    }
    ~NeuralFixture() { std::filesystem::remove_all(directory); }
    std::unique_ptr<PotMTPNN1> make(int layers, bool sigmoid = false) {
        GPSettings settings;
        settings.Modify("mtpcoeff",(directory/"model.mtp").string());
        settings.Modify("nncoeff",(directory/"absent.nn").string());
        settings.Modify("spcoeff",(directory/"absent.sc").string());
        settings.Modify("nnlay",std::to_string(layers));
        settings.Modify("nnhidden","4");
        settings.Modify("nnhidden2","3");
        auto model = std::make_unique<PotMTPNN1>(settings);
        std::vector<double> coefficients(model->nn_coeff_size);
        for (int i = 0; i < coefficients.size(); ++i) coefficients[i] = .65*std::sin(.7*(i+1));
        auto configure = [&](auto& network) {
            network.set_coeffs(coefficients.data());
            network.acfunc = sigmoid ? IMRMLPNN::SIGMOID : IMRMLPNN::TANH;
            for (int i = 0; i < network.n_input; ++i) {
                network.norm_mean[i] = .1*(i+1);
                network.norm_irstd[i] = .8+.2*i;
            }
        };
        if (layers == 1) configure(*model->p_nn); else configure(*model->p_nn2);
        model->species_coeffs = {-.3,.2};
        return model;
    }
};
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
Eigen::MatrixXd native_gradient_fd(AnyPotential& api, const Atoms& reference, double step) {
    const int dimension = 9+3*reference.size();
    const Eigen::MatrixXd fractional = [&] {
        Eigen::MatrixXd f(reference.size(),3);
        for (int i=0;i<reference.size();++i) f.row(i)=reference.pos_[i]*reference.cell_.inverse();
        return f;
    }();
    auto gradient = [&](const Eigen::VectorXd& q) {
        auto work = reference;
        for (int a=0;a<3;++a)
            for (int b=0;b<3;++b) work.cell_(a,b)+=q[3*a+b];
        for (int i=0;i<work.size();++i)
            work.pos_[i]=fractional.row(i)*work.cell_+q.segment<3>(9+3*i).transpose();
        api.getEFS(work);
        const Eigen::Matrix3d inverse=work.cell_.inverse();
        Eigen::Matrix3d cell_gradient=-inverse.transpose()*work.stress_.transpose();
        Eigen::VectorXd result(dimension);
        for (int i=0;i<work.size();++i) {
            // EFS virial refers to the current fractional positions. The
            // native chart holds the reference f fixed while residual u varies.
            const Eigen::RowVector3d delta_f=q.segment<3>(9+3*i).transpose()*inverse;
            cell_gradient+=delta_f.transpose()*work.forces_[i].transpose();
            result.segment<3>(9+3*i)=-work.forces_[i];
        }
        for (int a=0;a<3;++a)
            for (int b=0;b<3;++b) result[3*a+b]=cell_gradient(a,b);
        return result;
    };
    Eigen::MatrixXd result(dimension,dimension);
    Eigen::VectorXd q=Eigen::VectorXd::Zero(dimension);
    for (int column=0;column<dimension;++column) {
        q[column]=step;const auto positive=gradient(q);
        q[column]=-step;const auto negative=gradient(q);q[column]=0.;
        result.col(column)=(positive-negative)/(2*step);
    }
    return result;
}
void same_efs(const Atoms& actual, const Atoms& expected, double factor = 1.) {
    REQUIRE(std::abs(actual.energy_-factor*expected.energy_) < 1e-11);
    REQUIRE((actual.stress_-factor*expected.stress_).norm() < 1e-11);
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE((actual.forces_[i]-factor*expected.forces_[i]).norm() < 1e-11);
}
}

TEST_CASE("Legacy NN1 and NN2 curvature retains neural and moment chain terms",
          "[mtpnn][hessian][cell][hvp]") {
    const int layers = GENERATE(1,2);
    const bool sigmoid = GENERATE(false,true);
    const bool partial = GENERATE(false,true);
    INFO("layers=" << layers << " sigmoid=" << sigmoid << " partial=" << partial);
    NeuralFixture fixture;
    auto owner = fixture.make(layers,sigmoid);
    auto& model = *owner;
    AnyPotential& api = model;
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_cell_hessian());
    REQUIRE(api.supports_analytical_hessian_vector_product());
    REQUIRE(api.supports_site_hessian());
    REQUIRE(api.supports_site_hvp());
    std::vector<double> coefficients(model.get_NL_coeffs_count());
    model.nl_get_coeffs(coefficients.data());
    auto atom = PairCellTest::geometry(partial), reference = atom;
    Neighborhoods nbhs(atom,api.cutoff);
    bool repeated = false;
    for (int i = 0; i < nbhs.size(); ++i) {
        std::map<int,int> counts;
        for (int source : nbhs[i].indices_) repeated |= ++counts[source] > 1;
    }
    REQUIRE(repeated);
    api.getEFS(reference,nbhs);
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    api.getEFSHessianCell(atom,nbhs,h,cell,cross);
    const auto dense = HessianUtils::to_dense(h);
    const auto native = PairCellTest::native_matrix(h,cell,cross);
    REQUIRE((dense-dense.transpose()).norm() < 1e-11);
    REQUIRE((cell-cell.transpose()).norm() < 1e-11);
    REQUIRE(cell.norm() > 1e-7);
    REQUIRE(cross.norm() > 1e-7);
    same_efs(atom,reference);
    for (int probe = 1; probe <= 2; ++probe) {
        Eigen::VectorXd direction(3*atom.size()), product;
        for (int i = 0; i < direction.size(); ++i) direction[i] = std::sin(.71*(i+1)*probe);
        const auto before = atom;
        api.applyHessianVectorProduct(atom,nbhs,direction,product);
        REQUIRE(atom.pos_ == before.pos_);
        same_efs(atom,before);
        REQUIRE((dense*direction-product).norm() < 1e-11);
        for (double step : {1e-5,5e-6}) {
            const auto expected = force_fd(api,atom,direction,step,api.cutoff);
            REQUIRE((product-expected).norm() < 2e-7*std::max(1.,expected.norm()));
        }
        auto alias = direction;
        api.applyHessianVectorProduct(atom,nbhs,alias,alias);
        REQUIRE((alias-product).norm() < 1e-12);
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(3*atom.size()), product;
        for (int i = 0; i < atom.size(); ++i) translation[3*i+axis] = 1.;
        api.applyHessianVectorProduct(atom,nbhs,translation,product);
        REQUIRE(product.isZero(1e-12));
        REQUIRE((dense*translation).norm() < 1e-11);
    }
    // Independently differentiate EFS forces and the native cell gradient
    // before consulting scalar-energy stencils. This checks both mixed block
    // orientations without relying on energy subtraction or extrapolation.
    for (double step : {2e-5,1e-5}) {
        const auto expected=native_gradient_fd(api,atom,step);
        INFO("native EFS gradient FD step=" << step);
        REQUIRE((native-expected).norm() < 5e-7*std::max(1.,expected.norm()));
    }
    // h=.001 is outside the observed asymptotic regime for the full-PBC NN
    // fixture. Use the two finer stencils, and verify their convergence before
    // applying Richardson rather than importing the coarse-stencil error.
    const auto energy_coarse = PairCellTest::energy_fd(api,atom,.0005,true);
    const auto energy_fine = PairCellTest::energy_fd(api,atom,.00025,true);
    const double coarse_error=(native-energy_coarse).norm(),fine_error=(native-energy_fine).norm();
    INFO("energy FD h=.0005 error=" << coarse_error << " h=.00025 error=" << fine_error);
    REQUIRE(fine_error < .35*coarse_error+2e-7);
    const Eigen::MatrixXd extrapolated=(4*energy_fine-energy_coarse)/3.;
    REQUIRE((native-extrapolated).norm() < 2e-6*std::max(1.,native.norm()));
    // Classical moment output coefficients do not enter a neural model.
    // Changing them catches an accidental nonvirtual call to the linear kernel.
    const double old = model.p_mtp->moment_coeffs[0];
    model.p_mtp->moment_coeffs[0] = 123.;
    auto changed = atom;
    api.getEFSHessian(changed,h);
    REQUIRE((HessianUtils::to_dense(h)-dense).norm() < 1e-12);
    model.p_mtp->moment_coeffs[0] = old;
    for (int route = 0; route < 7; ++route) {
        auto work = atom;
        if (route == 0) api.getEFSHessian(work,nbhs,h);
        if (route == 1) api.getEFSHessianOMP(work,nbhs,h);
        if (route == 2) api.getEFSHessianSymm(work,h);
        if (route == 3) api.getEFSHessianSymmOMP(work,h);
        if (route == 4) api.getEFSHessianCell(work,h,cell,cross);
        if (route == 5) api.getEFSHessianCellOMP(work,nbhs,h,cell,cross);
        if (route == 6) api.getEFSHessianCellSymmOMP(work,h,cell,cross);
        REQUIRE((HessianUtils::to_dense(h)-dense).norm() < 1e-11);
        same_efs(work,reference);
    }
    api.add_efs = true;
    api.getEFSHessianCellSymm(atom,h,cell,cross);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-native).norm() < 1e-11);
    same_efs(atom,reference,2.);
    std::vector<double> after(coefficients.size());
    model.nl_get_coeffs(after.data());
    REQUIRE(after == coefficients);
}

TEST_CASE("Legacy neural local HVP and Hessian agree with affine self image curvature",
          "[mtpnn][hessian][site][self-image]") {
    const int layers = GENERATE(1,2);
    NeuralFixture fixture;
    auto model = fixture.make(layers);
    AnyPotential& api = *model;
    Atoms atom;
    atom.cell_ = 3.2*Eigen::Matrix3d::Identity();
    atom.pos_ = {{.2,.3,.4}};
    atom.chemmap_ = {0};
    Neighborhoods nbhs(atom,api.cutoff);
    REQUIRE(nbhs[0].nbhcount_ > 1);
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    api.getEFSHessianCellSymm(atom,h,cell,cross);
    REQUIRE(h(0,0).isZero(1e-12));
    REQUIRE(cross.isZero(1e-12));
    REQUIRE(cell.norm() > 1e-7);
    for (double step : {.001,.0005}) {
        const auto expected = PairCellTest::energy_fd(api,atom,step,true);
        REQUIRE((PairCellTest::native_matrix(h,cell,cross)-expected).norm() < 100*step*step+2e-6);
    }
    double energy = 0., directional_energy = 0.;
    SiteCoordinateMatrix gradient,directional_gradient,product;
    api.evaluate_site_hessian(nbhs[0],energy,gradient,h);
    SiteCoordinateMatrix direction(nbhs[0].nbhcount_,3);
    for (int j = 0; j < direction.rows(); ++j)
        for (int a = 0; a < 3; ++a) direction(j,a) = std::sin(.7*(3*j+a+1));
    api.apply_site_hvp(nbhs[0],direction,directional_energy,directional_gradient,product);
    REQUIRE(gradient.rows() == nbhs[0].nbhcount_);
    REQUIRE(product.rows() == direction.rows());
    REQUIRE(energy == Catch::Approx(directional_energy));
    REQUIRE((gradient-directional_gradient).norm() < 1e-12);
    const Eigen::Map<const Eigen::VectorXd> flattened(direction.data(),direction.size());
    const Eigen::Map<const Eigen::VectorXd> flattened_product(product.data(),product.size());
    REQUIRE((HessianUtils::to_dense(h)*flattened-flattened_product).norm() < 1e-11);
    auto alias = direction;
    api.apply_site_hvp(nbhs[0],alias,directional_energy,directional_gradient,alias);
    REQUIRE((alias-product).norm() < 1e-11);
}

TEST_CASE("Legacy neural curvature rejects unvalidated branches and invalid directions",
          "[mtpnn][hessian][contract]") {
    NeuralFixture fixture;
    auto model = fixture.make(1);
    AnyPotential& api = *model;
    auto atom = PairCellTest::geometry();
    Neighborhoods nbhs(atom,api.cutoff);
    Eigen::VectorXd result = Eigen::VectorXd::Constant(2,123.);
    const auto previous = result;
    auto bad = Eigen::VectorXd::Zero(3*atom.size()+1).eval();
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom,nbhs,bad,result));
    REQUIRE(result == previous);
    bad = Eigen::VectorXd::Zero(3*atom.size());
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom,nbhs,bad,result));
    REQUIRE(result == previous);
    for (bool* correction : {&model->p_mtp->hc_correction,&model->p_mtp->rR_correction}) {
        *correction = true;
        REQUIRE_FALSE(api.supports_analytical_hessian());
        REQUIRE_FALSE(api.supports_analytical_hessian_vector_product());
        REQUIRE_FALSE(api.supports_analytical_cell_hessian());
        HessianMatrix h;
        REQUIRE_THROWS(api.getEFSHessian(atom,h));
        *correction = false;
    }
    model->p_nn->acfunc = IMRMLPNN::RELU;
    REQUIRE_FALSE(api.supports_site_hessian());
    REQUIRE_FALSE(api.supports_site_hvp());
    model->p_nn->acfunc = IMRMLPNN::TANH;
    model->nnlay = 3;
    REQUIRE_FALSE(api.supports_analytical_hessian());
}
