#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "unitest/les_curvature_fixture.h"
#include <fstream>
#include <limits>
#include <sstream>

using namespace Potentials;
namespace {
using namespace LESCurvature;
void check_model(AnyPotential& api, bool small) {
    REQUIRE(api.supports_analytical_hessian_vector_product());
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_cell_hessian());
    auto atom = geometry(small);
    Neighborhoods nbhs(atom,api.cutoff);
    api.getEFS(atom,nbhs);
    const auto before = atom;
    std::vector<double> initial(api.get_NL_coeffs_count()),after(initial.size());
    api.nl_get_coeffs(initial.data());
    Eigen::VectorXd direction(3*atom.size()),product;
    for (int i = 0; i < direction.size(); ++i) direction[i] = std::sin(.71*(i+1));
    api.applyHessianVectorProduct(atom,nbhs,direction,product);
    REQUIRE(product.allFinite());
    REQUIRE(atom.pos_ == before.pos_); REQUIRE(atom.cell_ == before.cell_);
    REQUIRE(atom.energy_ == before.energy_); REQUIRE(atom.forces_ == before.forces_);
    REQUIRE(atom.stress_ == before.stress_); REQUIRE(atom.charges_ == before.charges_);
    REQUIRE(atom.site_energies_ == before.site_energies_);
    for (double step : {2e-5,1e-5}) {
        const auto expected = force_fd(api,atom,direction,step,api.cutoff);
        INFO("step=" << step << " HVP norm=" << expected.norm());
        REQUIRE((product-expected).norm() < 3e-6*std::max(1.,expected.norm()));
    }
    // Independent energy oracle checks the complete directional curvature.
    for (double step : {.001,.0005}) {
        auto plus = atom,minus = atom,center = atom;
        for (int i = 0; i < atom.size(); ++i) {
            plus.pos_[i] += step*direction.segment<3>(3*i).transpose();
            minus.pos_[i] -= step*direction.segment<3>(3*i).transpose();
        }
        api.getEFS(plus); api.getEFS(minus); api.getEFS(center);
        const double expected = (plus.energy_+minus.energy_-2*center.energy_)/(step*step);
        REQUIRE(direction.dot(product) == Catch::Approx(expected).margin(2e-4*std::max(1.,std::abs(expected))));
    }
    auto alias = direction;
    api.applyHessianVectorProduct(atom,nbhs,alias,alias);
    REQUIRE((alias-product).norm() < 1e-11);
    api.add_efs = true;
    Eigen::VectorXd additive;
    api.applyHessianVectorProduct(atom,nbhs,direction,additive);
    api.add_efs = false;
    REQUIRE((additive-product).norm() < 1e-11);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(direction.size()),zero;
        for (int i = 0; i < atom.size(); ++i) translation[3*i+axis] = 1.;
        api.applyHessianVectorProduct(atom,nbhs,translation,zero);
        REQUIRE(zero.norm() < 1e-10);
    }
    // Self-adjointness probes different mixed position/charge contractions.
    Eigen::VectorXd other = Eigen::VectorXd::LinSpaced(direction.size(),-.4,.7),other_product;
    api.applyHessianVectorProduct(atom,nbhs,other,other_product);
    REQUIRE(other.dot(product) == Catch::Approx(direction.dot(other_product)).margin(1e-10));
    api.nl_get_coeffs(after.data()); REQUIRE(initial == after);
    auto invalid = direction; invalid[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom,nbhs,invalid,additive));
}
}

TEST_CASE("LES reciprocal Cartesian response differentiates fixed and changing charges",
          "[les][sog][hvp][reciprocal]") {
    const bool sog = GENERATE(false,true);
    const bool self = GENERATE(false,true);
    const bool zero = GENERATE(false,true);
    const bool recurrence = GENERATE(false,true);
    const bool changing = GENERATE(false,true);
    LES engine; configure_engine(engine,sog,self);
    engine.sog_include_zero = zero;
    engine.use_recurrence = recurrence;
    engine.use_cache = recurrence;
    const auto atom = geometry();
    Eigen::VectorXd q(4),qd = Eigen::VectorXd::Zero(4);
    q << .31,-.17,.23,.11;
    if (changing) qd << -.13,.07,.19,-.11;
    SiteCoordinateMatrix velocity(4,3);
    for (int i = 0; i < 4; ++i)
        for (int a = 0; a < 3; ++a) velocity(i,a) = std::sin(.7*(3*i+a+1));
    const auto response = engine.evaluate_cartesian_directional(atom,q,qd,velocity);
    const auto base = engine.evaluate(atom,q);
    REQUIRE((response.dEdq-base.dEdq).norm() < 1e-11);
    for (double step : {2e-5,1e-5}) {
        auto plus = atom,minus = atom;
        for (int i = 0; i < 4; ++i) {
            plus.pos_[i] += step*velocity.row(i);
            minus.pos_[i] -= step*velocity.row(i);
        }
        const auto ep = engine.evaluate(plus,q+step*qd),em = engine.evaluate(minus,q-step*qd);
        REQUIRE((response.force_directional-(ep.forces-em.forces)/(2*step)).norm() < 2e-7);
        REQUIRE((response.dEdq_directional-(ep.dEdq-em.dEdq)/(2*step)).norm() < 2e-7);
    }
}

TEST_CASE("Ordinary LES Cartesian HVP includes all latent charge chain terms",
          "[les][sog][hvp][mtp]") {
    const bool sog = GENERATE(false,true);
    const bool small = GENERATE(false,true);
    PotMTP_LES model("unitest/test-data/portable-mtp/les.mtp");
    initialize(model,sog);
    check_model(model,small);
}

TEST_CASE("MTP-M Coulomb and multichannel SOG HVP retain model-aware short range curvature",
          "[les][sog][hvp][mtp_m]") {
    const bool sog = GENERATE(false,true);
    const bool neural = GENERATE(false,true);
    if (neural) {
        PotMTP_M<m_model_nn1> model;
        initialize(model,sog);
        check_model(model,false);
    } else {
        PotMTP_M<m_model_mtpq2> model;
        initialize(model,sog);
        check_model(model,true);
    }
}
