#include <catch2/catch_all.hpp>
#include "unitest/les_curvature_fixture.h"
#include <array>
#include <thread>

using namespace Potentials;
using namespace LESCurvature;
using Catch::Approx;
namespace {
void compare_evaluation(const Atoms& actual,const Atoms& expected) {
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(2e-11));
    REQUIRE((actual.stress_-expected.stress_).norm()<2e-10);
    for(int i=0;i<actual.size();++i) {
        REQUIRE((actual.forces_[i]-expected.forces_[i]).norm()<2e-10);
        REQUIRE(actual.site_energies_[i]==Approx(expected.site_energies_[i]).margin(2e-11));
    }
    REQUIRE(actual.charges_.size()==expected.charges_.size());
    for(int i=0;i<actual.charges_.size();++i) REQUIRE(actual.charges_[i]==Approx(expected.charges_[i]).margin(1e-13));
}
// The reference deliberately uses the pre-existing base charge-force graph
// and the legacy reciprocal API, rather than the new workspace composition.
template<class T> Atoms independently_composed(PotMTP_M<T>& reference,Atoms atom) {
    Neighborhoods nbhs(atom,reference.cutoff);
    reference.lr_les=false;reference.getEFS(atom,nbhs);reference.lr_les=true;
    const int channels=reference.les_channels();
    Eigen::MatrixXd q(atom.size(),channels);
    for(int i=0;i<atom.size();++i) {
        reference.CalcSiteBasis(nbhs[i]);
        for(int c=0;c<channels;++c) {
            q(i,c)=reference.les_species[c*reference.species_count+nbhs[i].centertype];
            for(int a=0;a<reference.alpha_scalar_moments;++a)
                q(i,c)+=reference.les_moment[c*reference.alpha_scalar_moments+a]
                    *reference.moment_vals[reference.alpha_moment_mapping[a]];
        }
    }
    atom.charges_.clear();
    for(int c=0;c<channels;++c) {
        LES::ChannelScope selected(reference.model_les,c);
        const auto response=reference.model_les.evaluate(atom,q.col(c));
        atom.energy_+=response.energy;atom.stress_+=response.virial;
        for(int i=0;i<atom.size();++i) {
            atom.forces_[i]+=response.forces.row(i);atom.site_energies_[i]+=response.per_atom_energy[i];
            reference.add_latent_charge_forces(i,nbhs[i],response.dEdq[i],
                reference.les_moment+c*reference.alpha_scalar_moments,atom);
        }
    }
    if(!reference.uses_sog()) for(int i=0;i<atom.size();++i) atom.charges_.push_back(q(i,0));
    return atom;
}
struct Inputs {
    Atoms atom;
    Eigen::Matrix<double,-1,3> forces;
    Eigen::Matrix3d stress;
    Eigen::VectorXd direction;
    explicit Inputs(int variant=0):atom(geometry()) {
        if(variant) {atom.cell_(0,0)+=.37;atom.pos_[1][1]+=.19;}
        forces.resize(atom.size(),3);direction.resize(3*atom.size());
        for(int i=0;i<forces.size();++i) forces.data()[i]=.1*std::sin(.7*(i+1));
        for(int i=0;i<direction.size();++i) direction[i]=.2*std::cos(.4*(i+1));
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) stress(a,b)=.03*std::cos(3*a+b+1.);
    }
};
template<class T> double evaluation_loss(PotMTP_M<T>& p,Inputs inputs,
    typename PotMTP_M<T>::EvaluationWorkspace& workspace) {
    Neighborhoods nbhs(inputs.atom,p.cutoff);p.getEFS(inputs.atom,nbhs,workspace);
    double result=.17*inputs.atom.energy_+(inputs.stress.array()*inputs.atom.stress_.array()).sum();
    for(int i=0;i<inputs.atom.size();++i) result+=inputs.forces.row(i).dot(inputs.atom.forces_[i]);
    return result;
}
struct AllResponses {Atoms atom;Eigen::VectorXd gradient,product;};
template<class T> AllResponses all_responses(PotMTP_M<T>& p,const Inputs& inputs,
    typename PotMTP_M<T>::EvaluationWorkspace& workspace) {
    AllResponses result;result.atom=inputs.atom;Neighborhoods nbhs(result.atom,p.cutoff);
    p.getEFS(result.atom,nbhs,workspace);
    result.gradient=Eigen::VectorXd::Zero(p.get_NL_coeffs_count());
    p.AccumulateEFSCombinationGard(result.atom,nbhs,.17,inputs.forces,inputs.stress,result.gradient.data(),workspace);
    p.applyHessianVectorProduct(result.atom,nbhs,inputs.direction,result.product,workspace);
    return result;
}
void compare_responses(const AllResponses& actual,const AllResponses& expected) {
    compare_evaluation(actual.atom,expected.atom);
    REQUIRE((actual.gradient-expected.gradient).norm()<2e-10);
    REQUIRE((actual.product-expected.product).norm()<2e-10);
}
}

TEMPLATE_TEST_CASE("MTP-M LES evaluation workspaces preserve independently composed EFS and full parameter gradients",
    "[mtp_m][les][evaluation-workspace]",m_model_mtpli,m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    const bool sog=GENERATE(false,true);
    PotMTP_M<TestType> p,reference;initialize(p,sog);initialize(reference,sog);
    Inputs inputs;auto workspace=p.make_evaluation_workspace();
    const auto actual=all_responses(p,inputs,workspace);
    compare_evaluation(actual.atom,independently_composed(reference,inputs.atom));
    Eigen::VectorXd coefficients(p.get_NL_coeffs_count());p.nl_get_coeffs(coefficients.data());
    std::vector<int> samples={0,p.radial_coeff_size+p.species_coeff_size,p.les_fitting_offset(),
        p.les_fitting_offset()+p.les_channels()*p.species_count};
    if(p.model_coeff_size) samples.push_back(p.les_fitting_offset()-1);
    if(sog) {samples.push_back(p.sog_fitting_offset());samples.push_back(p.sog_fitting_offset()+p.model_les.sog_weights.size());}
    for(double step : {2e-5,6e-6}) {
        CAPTURE(sog,step);
        for(int sample:samples) {
            CAPTURE(sample);
            auto plus=coefficients,minus=coefficients;plus[sample]+=step;minus[sample]-=step;
            p.nl_set_coeffs(plus.data());const double ep=evaluation_loss(p,inputs,workspace);
            p.nl_set_coeffs(minus.data());const double em=evaluation_loss(p,inputs,workspace);
            p.nl_set_coeffs(coefficients.data());
            REQUIRE(actual.gradient[sample]==Approx((ep-em)/(2*step)).margin(3e-6));
        }
        auto plus=inputs.atom,minus=inputs.atom;
        for(int i=0;i<plus.size();++i) {
            plus.pos_[i]+=step*inputs.direction.segment<3>(3*i).transpose();
            minus.pos_[i]-=step*inputs.direction.segment<3>(3*i).transpose();
        }
        Neighborhoods np(plus,p.cutoff),nm(minus,p.cutoff);
        p.getEFS(plus,np,workspace);p.getEFS(minus,nm,workspace);
        Eigen::VectorXd finite(inputs.direction.size());
        for(int i=0;i<plus.size();++i) finite.segment<3>(3*i)=-(plus.forces_[i]-minus.forces_[i]).transpose()/(2*step);
        REQUIRE((actual.product-finite).norm()<3e-6*(1.+finite.norm()));
    }
    compare_responses(all_responses(p,inputs,workspace),actual);
}

TEMPLATE_TEST_CASE("MTP-M structure workspaces isolate multichannel caches and responses across threads",
    "[mtp_m][les][evaluation-workspace][threads]",m_model_mtpq2,m_model_nn1,m_model_nn2) {
    const bool sog=GENERATE(false,true);
    PotMTP_M<TestType> p;initialize(p,sog);
    p.model_les.sog_channel=sog ? 1 : 0;
    Inputs first,second(1);
    auto legacy=first.atom;p.getEFS(legacy);
    const auto diagnostics=p.LES_all_latents;const auto charges=p.LES_all_qs;
    const auto grid=p.model_les.kvecs;const auto kernel=p.model_les.k_kernel;
    REQUIRE_FALSE(grid.empty());
    const auto& serial=p.serial_evaluation_workspace_.reciprocal;
    REQUIRE(grid==serial.kvecs);REQUIRE(kernel==serial.k_kernel);
    REQUIRE(p.model_les.kvec_factors==serial.kvec_factors);
    REQUIRE(p.model_les.k_kernel_derivative==serial.k_kernel_derivative);
    REQUIRE(p.model_les.reciprocal_taper()==serial.grid_taper_);
    REQUIRE(p.model_les.reciprocal_taper_derivative()==serial.grid_taper_derivative_);
    LES::Workspace independent;
    const int last=p.les_channels()-1;
    p.model_les.evaluate(legacy,p.LES_all_latents.col(last),independent,last);
    REQUIRE(grid==independent.kvecs);REQUIRE(kernel==independent.k_kernel);
    REQUIRE(p.model_les.kvec_factors==independent.kvec_factors);
    REQUIRE(p.model_les.k_kernel_derivative==independent.k_kernel_derivative);
    REQUIRE(p.model_les.reciprocal_taper()==independent.grid_taper_);
    REQUIRE(p.model_les.reciprocal_taper_derivative()==independent.grid_taper_derivative_);
    const int selected=p.model_les.sog_channel;
    Eigen::VectorXd coefficients(p.get_NL_coeffs_count());p.nl_get_coeffs(coefficients.data());
    auto one=p.make_evaluation_workspace(),two=p.make_evaluation_workspace();
    const std::array<AllResponses,2> expected={all_responses(p,first,one),all_responses(p,second,two)};
    std::array<AllResponses,2> actual;
    auto compute=[&](int slot) {
        auto workspace=p.make_evaluation_workspace();
        for(int repeat=0;repeat<4;++repeat) actual[slot]=all_responses(p,slot ? second : first,workspace);
    };
    std::thread a(compute,0),b(compute,1);a.join();b.join();
    compare_responses(actual[0],expected[0]);compare_responses(actual[1],expected[1]);
    auto threaded=first.atom;Neighborhoods nbhs(threaded,p.cutoff);p.getEFSOMPthread(threaded,nbhs);
    compare_evaluation(threaded,expected[0].atom);
    REQUIRE(p.model_les.sog_channel==selected);
    REQUIRE(p.model_les.kvecs==grid);REQUIRE(p.model_les.k_kernel==kernel);
    REQUIRE(p.LES_all_latents==diagnostics);REQUIRE(p.LES_all_qs==charges);
    Eigen::VectorXd after(coefficients.size());p.nl_get_coeffs(after.data());REQUIRE(after==coefficients);
}

TEST_CASE("MTP-M structure workspace reuses geometry storage and invalidates changed physical kernels",
    "[mtp_m][les][evaluation-workspace][reuse]") {
    PotMTP_M<m_model_nn2> p;initialize(p,true);
    Inputs first,second(1);auto workspace=p.make_evaluation_workspace();
    const auto expected=all_responses(p,first,workspace);
    const auto* latent_storage=workspace.latents.data();const auto* descriptors=workspace.descriptors.data();
    const auto* grid=workspace.reciprocal.cached_grid_.data();const auto* cosine=workspace.reciprocal.cosine.data();
    for(int repeat=0;repeat<3;++repeat) {
        compare_responses(all_responses(p,first,workspace),expected);
        REQUIRE(workspace.latents.data()==latent_storage);REQUIRE(workspace.descriptors.data()==descriptors);
        REQUIRE(workspace.reciprocal.cached_grid_.data()==grid);REQUIRE(workspace.reciprocal.cosine.data()==cosine);
    }
    auto fresh=p.make_evaluation_workspace();
    compare_responses(all_responses(p,second,workspace),all_responses(p,second,fresh));
    p.model_les.sog_weights[0]+=.17;p.model_les.sog_log_bandwidths[0]+=.03;
    fresh=p.make_evaluation_workspace();
    compare_responses(all_responses(p,second,workspace),all_responses(p,second,fresh));
    p.model_les.use_cache=false;
    compare_responses(all_responses(p,second,workspace),all_responses(p,second,fresh));
}

TEST_CASE("MTP-M legacy diagnostic publication invalidates incompatible default LES cache",
    "[mtp_m][les][evaluation-workspace][diagnostics]") {
    PotMTP_M<m_model_mtpq2> p;initialize(p,true);
    Inputs first,second(1);auto base=first.atom;p.getEFS(base);
    const auto charges=p.LES_all_qs;
    // Seed the separate legacy LES cache with another cell, then publish the
    // MTP-M serial workspace diagnostics. This must not mix their taper/grid.
    p.model_les.evaluate(second.atom,charges);
    p.getEFS(base);
    LES::Workspace independent;
    const auto expected=p.model_les.evaluate(second.atom,charges,independent,p.model_les.sog_channel);
    const auto actual=p.model_les.evaluate(second.atom,charges);
    REQUIRE(actual.energy==Approx(expected.energy).margin(1e-13));
    REQUIRE((actual.forces-expected.forces).norm()<1e-13);
    REQUIRE((actual.virial-expected.virial).norm()<1e-13);
    REQUIRE(p.model_les.reciprocal_taper()==independent.grid_taper_);
    REQUIRE(p.model_les.reciprocal_taper_derivative()==independent.grid_taper_derivative_);
}
