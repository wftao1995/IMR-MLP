#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO3.h"
#include "descriptors/descriptor_BispectrumSO4.h"
#include "potentials/potential_descriptor_linear.h"
#include "unitest/pair_cell_fixture.h"

#include <limits>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace {
Neighborhood bispectrum_environment(bool polar, double transverse = 0., double pole_sign = 1.) {
    Neighborhood nbh;
    nbh.centertype = 0; nbh.nbhcount_ = 3;
    nbh.indices_ = {7,7,11}; nbh.chemmap_ = {0,1,0};
    nbh.pos_ = {{1.7,.2,.4},{-.5,1.8,.3},{.4,-.6,2.1}};
    if (polar) nbh.pos_[0] = {transverse,-.3*transverse,pole_sign*1.7};
    for (const auto &position : nbh.pos_) nbh.distances_.push_back(position.norm());
    return nbh;
}
DescriptorSiteMatrix original_gradient(Descriptors &descriptor, Neighborhood &nbh,
    const Eigen::VectorXd &weights) {
    Eigen::VectorXd features; DescriptorGradient gradient; ComputeOptions options;
    options.compute_derivative = true; options.gradient = &gradient;
    descriptor.compute(nbh,features,options);
    DescriptorSiteMatrix result = DescriptorSiteMatrix::Zero(nbh.nbhcount_,3);
    for (int feature = 0; feature < weights.size(); ++feature)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int axis = 0; axis < 3; ++axis) result(j,axis) += weights[feature]*gradient.grad(feature,j,axis);
    return result;
}
void check_bispectrum(Descriptors &descriptor, bool polar, double transverse = 0., double pole_sign = 1.) {
    auto nbh = bispectrum_environment(polar,transverse,pole_sign);
    REQUIRE(descriptor.supports_weighted_site_hvp());
    REQUIRE(descriptor.supports_weighted_site_hessian());
    const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(descriptor.feature_size(),-.037,.061);
    double value = 0., hvp_value = 0.; DescriptorSiteMatrix gradient, hvp_gradient, product;
    Eigen::MatrixXd hessian;
    descriptor.compute_weighted_site_hessian(nbh,weights,value,gradient,hessian);
    Eigen::VectorXd features; descriptor.compute(nbh,features);
    REQUIRE(value == Catch::Approx(weights.dot(features)).epsilon(1e-10).margin(1e-10));
    const auto first = original_gradient(descriptor,nbh,weights);
    REQUIRE((gradient-first).norm() < 1e-9*(1.+first.norm()));
    REQUIRE((hessian-hessian.transpose()).norm() < 1e-9*(1.+hessian.norm()));
    DescriptorSiteMatrix direction(3,3); direction << .2,-.1,.3,-.2,.4,.1,.3,.1,-.2;
    descriptor.compute_weighted_site_hvp(nbh,weights,direction,hvp_value,hvp_gradient,product);
    REQUIRE((gradient-hvp_gradient).norm() < 1e-12);
    REQUIRE((Eigen::Map<const Eigen::VectorXd>(product.data(),9)
        -hessian*Eigen::Map<const Eigen::VectorXd>(direction.data(),9)).norm() < 1e-10*(1.+product.norm()));
    // Two independent central-difference scales, using the pre-existing
    // descriptor gradient pipeline rather than the new weighted contraction.
    for (double step : {8e-5,2e-5}) {
        CAPTURE(step,polar);
        for (int j = 0; j < 3; ++j)
            for (int axis = 0; axis < 3; ++axis) {
                auto plus = nbh, minus = nbh;
                plus.pos_[j][axis] += step; minus.pos_[j][axis] -= step;
                plus.distances_[j] = plus.pos_[j].norm(); minus.distances_[j] = minus.pos_[j].norm();
                const DescriptorSiteMatrix finite = (original_gradient(descriptor,plus,weights)
                    -original_gradient(descriptor,minus,weights))/(2.*step);
                REQUIRE((hessian.col(3*j+axis)-Eigen::Map<const Eigen::VectorXd>(finite.data(),9)).norm()
                    < 2e-6*(1.+hessian.col(3*j+axis).norm()));
            }
        auto plus = nbh, minus = nbh;
        for (int j = 0; j < 3; ++j) {
            plus.pos_[j] += step*direction.row(j); minus.pos_[j] -= step*direction.row(j);
            plus.distances_[j] = plus.pos_[j].norm(); minus.distances_[j] = minus.pos_[j].norm();
        }
        Eigen::VectorXd fp,fm;
        descriptor.compute(plus,fp); descriptor.compute(minus,fm);
        const double finite_energy = (weights.dot(fp)-2.*value+weights.dot(fm))/(step*step);
        const double contraction = (direction.array()*product.array()).sum();
        const double roundoff = 50.*std::numeric_limits<double>::epsilon()*(1.+std::abs(value))/(step*step);
        REQUIRE(contraction == Catch::Approx(finite_energy).margin(2e-6*(1.+std::abs(contraction))+roundoff));
    }
    auto clone = descriptor.clone(); DescriptorSiteMatrix cloned;
    clone->compute_weighted_site_hvp(nbh,weights,direction,hvp_value,hvp_gradient,cloned);
    REQUIRE((cloned-product).norm() < 1e-12);
}
}

TEST_CASE("SO3 weighted curvature preserves radial and real-imaginary invariant layout",
          "[descriptor][bispectrum-curvature][so3]") {
    const int angular = GENERATE(1,2,3);
    const bool polar = GENERATE(false,true);
    Descriptor_BispectrumSO3 descriptor(angular,0.,4.,3);
    check_bispectrum(descriptor,polar);
}

TEST_CASE("SO4 weighted curvature preserves half-column and species/self-weight conventions",
          "[descriptor][bispectrum-curvature][so4]") {
    const int angular = GENERATE(1,2,3,4);
    const bool polar = GENERATE(false,true);
    GPSettings settings;
    settings.Modify("cutoff","4"); settings.Modify("twojmax",std::to_string(angular));
    settings.Modify("weights","0.7 1.3"); settings.Modify("selfweights","0.4 0.8");
    Descriptor_BispectrumSO4 descriptor(settings);
    check_bispectrum(descriptor,polar);
}

TEST_CASE("SO3 near both polar axes keeps energy and gradient curvature consistent",
          "[descriptor][bispectrum-curvature][so3][polar]") {
    const double pole_sign = GENERATE(-1.,1.);
    const double transverse = GENERATE(0.,1e-7,1e-4);
    CAPTURE(pole_sign,transverse);
    Descriptor_BispectrumSO3 descriptor(3,0.,4.,3);
    check_bispectrum(descriptor,true,transverse,pole_sign);
}

TEST_CASE("SO3 Cartesian polar recurrence joins the spherical kernel continuously",
          "[descriptor][bispectrum-curvature][so3][polar]") {
    const double pole_sign = GENERATE(-1.,1.);
    Descriptor_BispectrumSO3 descriptor(3,0.,4.,3);
    const auto weights = Eigen::VectorXd::LinSpaced(descriptor.feature_size(),-.037,.061).eval();
    // y=-.3*x in this fixture, and the switching cone is (x²+y²)/r²=1e-6.
    const double boundary = 1.7*std::sqrt(1e-6/(1.09*(1.-1e-6)));
    Eigen::VectorXd previous_features;
    DescriptorSiteMatrix previous_gradient;
    for (double side : {1.-1e-7,1.+1e-7}) {
        CAPTURE(pole_sign,side);
        auto nbh = bispectrum_environment(true,boundary*side,pole_sign);
        Eigen::VectorXd features; descriptor.compute(nbh,features);
        const auto gradient = original_gradient(descriptor,nbh,weights);
        double value; DescriptorSiteMatrix native_gradient,product;
        const DescriptorSiteMatrix direction = DescriptorSiteMatrix::Zero(3,3);
        descriptor.compute_weighted_site_hvp(nbh,weights,direction,value,native_gradient,product);
        REQUIRE(value == Catch::Approx(weights.dot(features)).epsilon(1e-10).margin(1e-10));
        REQUIRE((gradient-native_gradient).norm() < 1e-9*(1.+gradient.norm()));
        if (previous_features.size()) {
            REQUIRE((features-previous_features).norm() < 1e-8*(1.+features.norm()));
            REQUIRE((gradient-previous_gradient).norm() < 1e-8*(1.+gradient.norm()));
        }
        previous_features = features; previous_gradient = gradient;
    }
}

TEST_CASE("SO4 weighted curvature advertises only implemented cutoff branches",
          "[descriptor][bispectrum-curvature][contract]") {
    Descriptor_BispectrumSO4 unsupported(4.,2,false,-1.,CutoffFunctions::CT_POLY2);
    REQUIRE_FALSE(unsupported.supports_weighted_site_hvp());
    REQUIRE_FALSE(unsupported.supports_weighted_site_hessian());
    auto nbh = bispectrum_environment(false);
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(unsupported.feature_size());
    DescriptorSiteMatrix direction = DescriptorSiteMatrix::Zero(3,3);
    DescriptorSiteMatrix gradient = DescriptorSiteMatrix::Constant(1,3,19.);
    DescriptorSiteMatrix product = DescriptorSiteMatrix::Constant(1,3,23.);
    double value = 17.;
    REQUIRE_THROWS(unsupported.compute_weighted_site_hvp(nbh,weights,direction,value,gradient,product));
    REQUIRE(value == 17.); REQUIRE(gradient.isConstant(19.)); REQUIRE(product.isConstant(23.));
    Descriptor_BispectrumSO4 hard(4.,2,false,-1.,CutoffFunctions::CT_HARD);
    check_bispectrum(hard,false);
}

TEST_CASE("Bispectrum descriptor potentials assemble native atomic and periodic cell curvature",
          "[potential][descriptor_linear][bispectrum-curvature][cell]") {
    const auto backend = GENERATE(std::string("biso3"),std::string("biso4"));
    const bool alternate=GENERATE(false,true);
    GPSettings settings;
    settings.Modify("species_count","2"); settings.Modify("descriptor:dtype",backend);
    settings.Modify("descriptor:biso3_lmax","2"); settings.Modify("descriptor:biso3_rsize","2");
    settings.Modify("descriptor:biso3_rmax","4");
    settings.Modify("descriptor:biso4:twojmax","2"); settings.Modify("descriptor:biso4:cutoff","4");
    if(alternate) {
        settings.Modify("descriptor:biso3_real_harmonics","true");
        settings.Modify("descriptor:biso4:contraction_backend","snap");
        settings.Modify("descriptor:biso4:wigner_backend","spline");
        settings.Modify("descriptor:biso4:normalize_U","true");
    }
    Potentials::PotentialDescriptorLinear potential(settings); Potentials::AnyPotential &api = potential;
    REQUIRE(api.supports_analytical_hessian()); REQUIRE(api.supports_analytical_hessian_vector_product());
    const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(potential.get_NL_coeffs_count(),-.01,.02);
    potential.nl_set_coeffs(weights.data());
    auto atom = PairCellTest::geometry(true); Potentials::HessianMatrix hessian;
    Eigen::MatrixXd cell,mixed; api.getEFSHessianCell(atom,hessian,cell,mixed);
    const auto actual = PairCellTest::native_matrix(hessian,cell,mixed);
    for (double step : {4e-4,2e-4}) {
        const auto finite = PairCellTest::energy_fd(api,atom,step,true);
        REQUIRE((actual-finite).cwiseAbs().maxCoeff() < 1e-4*(1.+actual.cwiseAbs().maxCoeff()));
    }
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(3*atom.size(),-.2,.3);
    Eigen::VectorXd product; api.applyHessianVectorProduct(atom,direction,product);
    REQUIRE((product-HessianUtils::to_dense(hessian)*direction).norm() < 1e-9*(1.+product.norm()));
    if(alternate) {
        const auto path=std::filesystem::temp_directory_path()/("so4-real-roundtrip-"+backend+std::to_string(getpid())+".ini");
        struct Cleanup {std::filesystem::path p;~Cleanup(){std::filesystem::remove(p);}} cleanup{path};
        {std::ofstream output(path);potential.Save(output);}
        Potentials::PotentialDescriptorLinear restored(path.string());
        Potentials::HessianMatrix h;Eigen::MatrixXd c,m;
        restored.getEFSHessianCell(atom,h,c,m);
        REQUIRE((PairCellTest::native_matrix(h,c,m)-actual).norm()<1e-10*(1.+actual.norm()));
    }
}

TEST_CASE("SO4 runtime embeddings and radial maps share first and second derivatives",
          "[descriptor][bispectrum-curvature][so4][runtime]") {
    const auto convention=GENERATE("q1","q2","q3","q4");
    const auto mapping=GENERATE("linear","atan");
    const bool normalized=GENERATE(false,true);
    GPSettings settings;
    settings.Modify("cutoff","4");settings.Modify("twojmax","3");
    settings.Modify("quaternion_convention",convention);settings.Modify("radial_angle_map",mapping);
    settings.Modify("normalize_U",normalized?"true":"false");
    settings.Modify("weights","0.7 1.3");settings.Modify("selfweights","0.4 0.8");
    Descriptor_BispectrumSO4 descriptor(settings);
    check_bispectrum(descriptor,false);
}
TEST_CASE("SO4 U normalization scales every self and neighbor block consistently",
          "[descriptor][so4][normalization]") {
    GPSettings settings;settings.Modify("cutoff","4");settings.Modify("twojmax","4");
    Descriptor_BispectrumSO4 raw(settings);
    settings.Modify("normalize_U","true");Descriptor_BispectrumSO4 normalized(settings);
    auto site=bispectrum_environment(false);Eigen::VectorXd a,b;
    raw.compute(site,a);normalized.compute(site,b);
    for (int f=0;f<a.size();++f) {
        const auto j=raw.get_bi_ind(f);
        REQUIRE(b[f]==Catch::Approx(a[f]/std::sqrt(double((j[0]+1)*(j[1]+1)*(j[2]+1)))).margin(1e-11));
    }
}

TEST_CASE("SO4 rejects nonfinite construction and radial sign-branch derivatives",
          "[descriptor][so4][runtime][boundary]") {
    GPSettings s;s.Modify("cutoff","4");s.Modify("twojmax","2");s.Modify("radius","inf");
    REQUIRE_THROWS(Descriptor_BispectrumSO4(s));
    s.Modify("radius","2");s.Modify("quaternion_convention","q1");
    Descriptor_BispectrumSO4 d(s);Neighborhood site;
    site.nbhcount_=1;site.centertype=0;site.indices_={1};site.chemmap_={0};site.pos_={{2.,0.,0.}};site.distances_={2.};
    Eigen::VectorXd values;d.compute(site,values);REQUIRE(values.allFinite());
    DescriptorGradient g;ComputeOptions o;o.compute_derivative=true;o.gradient=&g;
    REQUIRE_THROWS_WITH(d.compute(site,values,o),Catch::Matchers::ContainsSubstring("sign-branch"));
    double e;DescriptorSiteMatrix gradient,product;
    REQUIRE_THROWS_WITH(d.compute_weighted_site_hvp(site,Eigen::VectorXd::Ones(d.size()),
        DescriptorSiteMatrix::Zero(1,3),e,gradient,product),Catch::Matchers::ContainsSubstring("sign-branch"));
}
