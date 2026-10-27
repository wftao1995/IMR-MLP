#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_Behler.h"

TEST_CASE("Behler zero angular exponent has finite value gradient and curvature at collinearity", "[descriptors][behler][curvature]") {
    const double sign = GENERATE(-1., 1.);
    GPSettings settings;
    settings.Modify("dispcount", "2"); settings.Modify("dispselect", "4 5");
    settings.Modify("cutoff", "5"); settings.Modify("eta", ".2");
    settings.Modify("zeta", "0"); settings.Modify("lambda", std::to_string(-sign));
    DescriptorBehler descriptor(settings);
    Neighborhood nbh; nbh.nbhcount_ = 2; nbh.pos_ = {{1, 0, 0}, {2 * sign, 0, 0}};
    nbh.distances_ = {1, 2}; nbh.chemmap_ = {0, 1}; nbh.centertype = 0;
    Eigen::VectorXd values; DescriptorGradient jacobian;
    ComputeOptions options; options.compute_derivative = true; options.gradient = &jacobian;
    descriptor.compute(nbh, values, options);
    REQUIRE(values.allFinite());
    for (int k = 0; k < jacobian.grad.size(); ++k) REQUIRE(std::isfinite(jacobian.grad.data()[k]));
    for (double step : {1e-4, 1e-5}) for (int i = 0; i < 2; ++i) for (int a = 0; a < 3; ++a) {
        auto plus = nbh, minus = nbh;
        plus.pos_[i][a] += step; minus.pos_[i][a] -= step;
        plus.distances_[i] = plus.pos_[i].norm(); minus.distances_[i] = minus.pos_[i].norm();
        Eigen::VectorXd vp, vm; descriptor.compute(plus, vp); descriptor.compute(minus, vm);
        for (int f = 0; f < 2; ++f)
            REQUIRE(jacobian.grad(f, i, a) == Catch::Approx((vp[f] - vm[f]) / (2 * step)).margin(1e-8));
    }
    Eigen::VectorXd weights(2); weights << .4, -.7;
    double value; DescriptorSiteMatrix gradient; Eigen::MatrixXd hessian;
    descriptor.compute_weighted_site_hessian(nbh, weights, value, gradient, hessian);
    REQUIRE(value == Catch::Approx(weights.dot(values)));
    REQUIRE(gradient.allFinite()); REQUIRE(hessian.allFinite());
    for (int i = 0; i < 2; ++i) for (int a = 0; a < 3; ++a)
        REQUIRE(gradient(i, a) == Catch::Approx(weights[0] * jacobian.grad(0, i, a) +
            weights[1] * jacobian.grad(1, i, a)).margin(1e-12));
}

TEST_CASE("Behler requires explicit species weighting and retains its default", "[descriptors][behler]") {
    GPSettings settings; settings.Modify("dispcount", "2"); settings.Modify("dispselect", "2 5"); settings.Modify("cutoff", "5");
    DescriptorBehler default_descriptor(settings);
    settings.Modify("weighted", "false"); DescriptorBehler explicit_descriptor(settings);
    Neighborhood nbh; nbh.nbhcount_ = 2; nbh.pos_ = {{1, 0, 0}, {0, 2, 0}};
    nbh.distances_ = {1, 2}; nbh.chemmap_ = {0, 1}; nbh.centertype = 0;
    Eigen::VectorXd before, after;
    default_descriptor.compute(nbh, before); explicit_descriptor.compute(nbh, after);
    REQUIRE(before.isApprox(after, 0.));
    settings.Modify("weighted", "true");
    REQUIRE_THROWS_WITH(DescriptorBehler{settings}, Catch::Matchers::ContainsSubstring("weighted=true requires weights or elements"));
}
