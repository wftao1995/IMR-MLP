#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptor_Behler.h"
#include "potentials/potential_descriptor_linear.h"
#include "unitest/pair_cell_fixture.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace {
class UnsupportedCurvatureDescriptor : public Descriptors {
public:
    void compute(Neighborhood &, Eigen::VectorXd &value) override { value.setZero(1); }
    int feature_size() const override { return 1; }
    std::unique_ptr<Descriptors> clone() const override {
        return std::make_unique<UnsupportedCurvatureDescriptor>();
    }
};
GPSettings behler_settings() {
    GPSettings settings;
    settings.Modify("dispcount", "5");
    settings.Modify("dispselect", "1 2 3 4 5");
    settings.Modify("cutoff", "5");
    settings.Modify("eta", "0.2");
    settings.Modify("rshift", "0.3");
    settings.Modify("kppa", "0.7");
    settings.Modify("zeta", "2");
    settings.Modify("lambda", "1");
    return settings;
}
GPSettings potential_settings(bool mtm) {
    GPSettings settings;
    settings.Modify("descriptor:dtype", mtm ? "mtm" : "behler");
    if (mtm) settings.Modify("descriptor:mtm:potfile", "unitest/test-data/portable-mtp/les.mtp");
    else for (const auto &[name, value] : behler_settings().settings)
        settings.Modify("descriptor:behler:" + name, value);
    settings.Modify("species_count", "2");
    return settings;
}
Neighborhood environment() {
    Neighborhood nbh;
    nbh.centertype = 0;
    nbh.nbhcount_ = 3;
    nbh.indices_ = {7, 7, 11};
    nbh.chemmap_ = {0, 1, 0};
    nbh.pos_ = {{2.2, .3, .4}, {-.6, 2.5, .2}, {.5, -.4, 2.8}};
    for (const auto &position : nbh.pos_) nbh.distances_.push_back(position.norm());
    return nbh;
}
DescriptorSiteMatrix contracted_gradient(Descriptors &descriptor, Neighborhood &nbh,
    const Eigen::VectorXd &weights) {
    DescriptorGradient features;
    ComputeOptions options;
    options.compute_derivative = true;
    options.gradient = &features;
    Eigen::VectorXd values;
    descriptor.compute(nbh, values, options);
    DescriptorSiteMatrix result = DescriptorSiteMatrix::Zero(nbh.nbhcount_, 3);
    for (int f = 0; f < weights.size(); ++f)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; ++a) result(j, a) += weights[f] * features.grad(f, j, a);
    return result;
}
void check_descriptor(Descriptors &descriptor) {
    auto nbh = environment();
    const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(descriptor.feature_size(), -.3, .7);
    REQUIRE(descriptor.supports_weighted_site_hvp());
    REQUIRE(descriptor.supports_weighted_site_hessian());
    double value = 0., hvp_value = 0.;
    DescriptorSiteMatrix gradient, hvp_gradient, product;
    Eigen::MatrixXd hessian;
    descriptor.compute_weighted_site_hessian(nbh, weights, value, gradient, hessian);
    Eigen::VectorXd features;
    descriptor.compute(nbh, features);
    REQUIRE(value == Catch::Approx(weights.dot(features)).margin(1e-12));
    REQUIRE((gradient - contracted_gradient(descriptor, nbh, weights)).norm() < 1e-10);
    REQUIRE((hessian - hessian.transpose()).norm() < 1e-12);
    DescriptorSiteMatrix direction(3, 3);
    direction << .3, -.2, .1, -.4, .5, .2, .1, .2, -.6;
    descriptor.compute_weighted_site_hvp(nbh, weights, direction, hvp_value, hvp_gradient, product);
    REQUIRE(hvp_value == Catch::Approx(value).margin(1e-12));
    REQUIRE((hvp_gradient - gradient).norm() < 1e-12);
    REQUIRE((Eigen::Map<const Eigen::VectorXd>(product.data(), 9)
            -hessian * Eigen::Map<const Eigen::VectorXd>(direction.data(), 9)).norm() < 1e-10);
    constexpr double step = 1e-5;
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) {
            auto plus = nbh, minus = nbh;
            plus.pos_[k][c] += step; minus.pos_[k][c] -= step;
            plus.distances_[k] = plus.pos_[k].norm();
            minus.distances_[k] = minus.pos_[k].norm();
            const auto gp = contracted_gradient(descriptor, plus, weights);
            const auto gm = contracted_gradient(descriptor, minus, weights);
            const DescriptorSiteMatrix difference = (gp - gm) / (2.*step);
            REQUIRE((hessian.col(3*k+c) - Eigen::Map<const Eigen::VectorXd>(difference.data(), 9)).norm() < 2e-7);
        }
    auto clone = descriptor.clone();
    DescriptorSiteMatrix cloned_product;
    clone->compute_weighted_site_hvp(nbh, weights, direction, hvp_value, hvp_gradient, cloned_product);
    REQUIRE((cloned_product - product).norm() < 1e-12);
}
}

TEST_CASE("Descriptor weighted curvature requires explicit support and validates input sizes",
          "[descriptor][weighted-curvature][contract]") {
    UnsupportedCurvatureDescriptor unsupported;
    REQUIRE_FALSE(unsupported.supports_weighted_site_hvp());
    REQUIRE_FALSE(unsupported.supports_weighted_site_hessian());
    auto nbh = environment();
    double value = 17.;
    DescriptorSiteMatrix gradient = DescriptorSiteMatrix::Constant(1, 3, 19.);
    DescriptorSiteMatrix product = DescriptorSiteMatrix::Constant(1, 3, 23.);
    DescriptorSiteMatrix direction = DescriptorSiteMatrix::Zero(3, 3);
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Constant(1, 1, 29.);
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(1);
    REQUIRE_THROWS(unsupported.compute_weighted_site_hvp(nbh, weights, direction, value, gradient, product));
    REQUIRE_THROWS(unsupported.compute_weighted_site_hessian(nbh, weights, value, gradient, hessian));
    DescriptorBehler behler(behler_settings());
    REQUIRE_THROWS(behler.compute_weighted_site_hvp(nbh, weights, direction, value, gradient, product));
    REQUIRE(value == 17.);
    REQUIRE(gradient.rows() == 1);
    REQUIRE(gradient.isConstant(19.));
    REQUIRE(product.rows() == 1);
    REQUIRE(product.isConstant(23.));
    REQUIRE(hessian.rows() == 1);
    REQUIRE(hessian(0, 0) == 29.);
}

TEST_CASE("Behler weighted curvature has finite Cartesian limits at collinear angles",
          "[descriptor][weighted-curvature][angular]") {
    auto settings = behler_settings();
    settings.Modify("zeta", "1");
    DescriptorBehler descriptor(settings);
    auto nbh = environment();
    nbh.nbhcount_ = 2;
    nbh.pos_ = {{2., 0., 0.}, {-2.2, 0., 0.}};
    nbh.distances_ = {2., 2.2};
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(5);
    weights[4] = 1.;
    double value = 0.;
    DescriptorSiteMatrix gradient;
    Eigen::MatrixXd hessian;
    descriptor.compute_weighted_site_hessian(nbh, weights, value, gradient, hessian);
    REQUIRE(std::isfinite(value));
    REQUIRE(gradient.allFinite());
    REQUIRE(hessian.allFinite());
    REQUIRE(hessian.norm() > 1e-8);
    auto plus = nbh, minus = nbh;
    plus.pos_[0][1] += 1e-5; minus.pos_[0][1] -= 1e-5;
    plus.distances_[0] = plus.pos_[0].norm(); minus.distances_[0] = minus.pos_[0].norm();
    const DescriptorSiteMatrix difference = (contracted_gradient(descriptor, plus, weights)
        -contracted_gradient(descriptor, minus, weights)) / 2e-5;
    REQUIRE((hessian.col(1) - Eigen::Map<const Eigen::VectorXd>(difference.data(), 6)).norm() < 1e-8);
}

TEST_CASE("MTM and Behler contract weighted curvature without feature Hessian tensors",
          "[descriptor][weighted-curvature]") {
    SECTION("MTM uses independent scratch and retains borrowed coefficients") {
        Potentials::PotMTP potential("unitest/test-data/portable-mtp/les.mtp");
        const std::vector<double> before(potential.regression_coeffs,
                                        potential.regression_coeffs + potential.reg_coeff_size);
        DescriptorMTM descriptor(potential);
        check_descriptor(descriptor);
        REQUIRE(std::vector<double>(potential.regression_coeffs,
                potential.regression_coeffs + potential.reg_coeff_size) == before);
        REQUIRE(potential.buff_site_hessian.hessian.empty());
        REQUIRE(potential.moment_vals == std::vector<double>(potential.alpha_moments_count, 0.));
    }
    SECTION("Behler G1 through G5 use their existing value and gradient conventions") {
        DescriptorBehler descriptor(behler_settings());
        check_descriptor(descriptor);
    }
}

TEST_CASE("Descriptor linear native curvature matches force differences and HVP",
          "[potential][descriptor_linear][weighted-curvature]") {
    const bool mtm = GENERATE(false, true);
    Potentials::PotentialDescriptorLinear potential(potential_settings(mtm));
    Potentials::AnyPotential &api = potential;
    const auto weights = Eigen::VectorXd::LinSpaced(potential.get_NL_coeffs_count(), -.13, .21).eval();
    potential.nl_set_coeffs(weights.data());
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_hessian_vector_product());
    REQUIRE(api.supports_analytical_cell_hessian());
    Atoms atom;
    atom.cell_ = 20. * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2., 2., 2.}, {4.2, 2.3, 2.4}, {2.5, 4.5, 2.2}};
    atom.chemmap_ = {0, 1, 0};
    auto ordinary = atom;
    api.getEFS(ordinary);
    Potentials::HessianMatrix hessian;
    api.getEFSHessian(atom, hessian);
    REQUIRE(atom.energy_ == Catch::Approx(ordinary.energy_).margin(1e-11));
    REQUIRE((atom.stress_ - ordinary.stress_).norm() < 1e-10);
    for (int i = 0; i < atom.size(); ++i) REQUIRE((atom.forces_[i] - ordinary.forces_[i]).norm() < 1e-10);
    const Eigen::MatrixXd dense = HessianUtils::to_dense(hessian);
    Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9, -.4, .6), product;
    api.applyHessianVectorProduct(atom, direction, product);
    REQUIRE((product - dense*direction).norm() < 1e-9);
    REQUIRE((dense - dense.transpose()).norm() < 1e-10);
    constexpr double step = 1e-5;
    for (int j = 0; j < atom.size(); ++j)
        for (int b = 0; b < 3; ++b) {
            auto plus = atom, minus = atom;
            plus.pos_[j][b] += step; minus.pos_[j][b] -= step;
            api.getEFS(plus); api.getEFS(minus);
            for (int i = 0; i < atom.size(); ++i)
                REQUIRE((hessian(i, j).col(b) + (plus.forces_[i] - minus.forces_[i])/(2.*step)).norm() < 2e-6);
        }
    Eigen::VectorXd translation = Eigen::VectorXd::Zero(9);
    for (int i = 0; i < 3; ++i) translation[3*i+1] = 1.;
    REQUIRE((dense * translation).norm() < 1e-10);
}

TEST_CASE("Descriptor linear native cell curvature retains periodic image geometry",
          "[potential][descriptor_linear][weighted-curvature][cell]") {
    const bool mtm = GENERATE(false, true);
    Potentials::PotentialDescriptorLinear potential(potential_settings(mtm));
    const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(potential.get_NL_coeffs_count(), -.03, .05);
    potential.nl_set_coeffs(weights.data());
    Potentials::AnyPotential &api = potential;
    auto atom = PairCellTest::geometry(true);
    Potentials::HessianMatrix hessian;
    Eigen::MatrixXd cell, mixed;
    api.getEFSHessianCell(atom, hessian, cell, mixed);
    const auto actual = PairCellTest::native_matrix(hessian, cell, mixed);
    const auto finite = PairCellTest::energy_fd(api, atom, 2e-4, true);
    REQUIRE((actual - finite).cwiseAbs().maxCoeff() < 3e-5);
}

TEST_CASE("MTM descriptor linear save preserves an absolute read-only template dependency",
          "[potential][descriptor_linear][weighted-curvature][io]") {
    const auto template_path = std::filesystem::absolute("unitest/test-data/portable-mtp/les.mtp").lexically_normal();
    std::ifstream original(template_path);
    const std::string before((std::istreambuf_iterator<char>(original)), {});
    Potentials::PotentialDescriptorLinear potential(potential_settings(true));
    const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(potential.get_NL_coeffs_count(), -.1, .2);
    potential.nl_set_coeffs(weights.data());
    const auto path = std::filesystem::temp_directory_path()
        / ("ut-mtm-weighted-" + std::to_string(getpid()) + ".ini");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove(path); } } cleanup{path};
    { std::ofstream output(path); potential.Save(output); }
    GPSettings serialized(path.string());
    REQUIRE(serialized["descriptor:mtm:potfile"] == template_path.string());
    Potentials::PotentialDescriptorLinear restored(path.string());
    std::vector<double> after(weights.size()); restored.nl_get_coeffs(after.data());
    REQUIRE(Eigen::Map<const Eigen::VectorXd>(after.data(), after.size()) == weights);
    std::ifstream unchanged(template_path);
    REQUIRE(std::string((std::istreambuf_iterator<char>(unchanged)), {}) == before);
}
