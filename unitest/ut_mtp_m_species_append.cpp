#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_m.h"
#include <sstream>

using namespace Potentials;
using Catch::Approx;
namespace {
std::string mtpm_append_fixture(bool split) {
    std::ostringstream out;
    out << "MTP\n"
        << "version = 1.1.0\n"
        << "potential_name = MTP1m\n"
        << "scaling = 1\n"
        << "species_count = 2\n"
        << "potential_tag =\n"
        << "radial_basis_split = " << (split ? 1 : 0) << "\n";
    if(split) out
        << "pair 0 0\n"
        << "radial_basis_type = RBChebyshev\n"
        << "  min_dist = 0.5\n"
        << "  max_dist = 4.0\n"
        << "  radial_basis_size = 2\n"
        << "pair 0 1\n"
        << "radial_basis_type = RBChebyCos\n"
        << "  min_dist = 0.0\n"
        << "  max_dist = 5.0\n"
        << "  radial_basis_size = 2\n"
        << "pair 1 0\n"
        << "radial_basis_type = RBChebyshevPoly5\n"
        << "  min_dist = 0.4\n"
        << "  max_dist = 4.5\n"
        << "  radial_basis_size = 2\n"
        << "  rb_b = -0.7\n"
        << "pair 1 1\n"
        << "radial_basis_type = RBChebyshev\n"
        << "  min_dist = 0.6\n"
        << "  max_dist = 3.2\n"
        << "  radial_basis_size = 2\n";
    else out<<"radial_basis_type = RBChebyshev\nmin_dist = 0.5\nmax_dist = 5.0\nradial_basis_size = 2\n";
    out
        << "radial_funcs_count = 1\n";
    if (true) {
        out << "radial_coeffs\n"
            << "  0-0\n    {0.70, -0.20}\n"
            << "  0-1\n    {0.45, 0.10}\n"
            << "  1-0\n    {-0.35, 0.25}\n"
            << "  1-1\n    {0.80, -0.15}\n";
    }
    out << "alpha_moments_count = 5\n"
        << "alpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, "
           "{0, 0, 1, 0}, {0, 0, 0, 1}}\n"
        << "alpha_index_times_count = 3\n"
        << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}\n"
        << "alpha_scalar_moments = 2\n"
        << "alpha_moment_mapping = {0, 4}\n";
    if (true) {
        out << "species_coeffs = {0.3, -0.1}\n"
            << "moment_coeffs = {1.2, -0.4}\n";
    }
    return out.str();
}

Atoms mtpm_append_geometry() {
    Atoms a;a.cell_=Eigen::Matrix3d::Identity()*12.;
    a.pos_={{1.,1.,1.},{2.1,1.3,1.2},{1.4,2.4,1.6}};
    a.chemmap_={0,1,0};return a;
}
void append_head_settings(GPSettings& settings) {
    settings.Modify("n_input","2");settings.Modify("n_neuron","3");settings.Modify("n_neuron2","2");
}
template<class T> void initialize_append(PotMTP_M<T>& p,bool split,int long_range=0) {
    GPSettings settings;append_head_settings(settings);p.model.init(settings);
    std::string source=mtpm_append_fixture(split);source.resize(source.find("species_coeffs"));
    if(long_range) {
        const int channels=long_range==2 ? 2 : 1;
        source+="les_species_coeffs = {0.23, -0.17";
        if(channels==2) source+=", 0.31, -0.08";
        source+="}\nles_moment_coeffs = {0.12, -0.03";
        if(channels==2) source+=", 0.04, 0.07";
        source+="}\n";
        if(long_range==2) {
            SOGModel definition;definition.channels=2;definition.weights={4.2,-1.3,3.1,-.8};
            definition.log_bandwidths={-.4,.2,-.2,.1};std::ostringstream metadata;definition.write(metadata);
            source+=metadata.str();
        } else p.lr_les=true;
    }
    std::istringstream in(source);p.Load(in);p.init_coeffs();
    for(int i=0;i<p.species_coeff_size;++i) p.species_coeffs[i]=.04*std::sin(i+1.);
    for(int i=0;i<p.moment_coeff_size;++i) p.moment_coeffs[i]=.06*std::cos(i+1.);
    for(int i=0;i<p.model_coeff_size;++i) p.model_coeffs[i]=.11*std::sin(.4*(i+1));
    p.train_init();
}
void append_same_efs(const Atoms& actual,const Atoms& expected) {
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(2e-11));
    REQUIRE((actual.stress_-expected.stress_).norm()<2e-10);
    for(int i=0;i<actual.size();++i) REQUIRE((actual.forces_[i]-expected.forces_[i]).norm()<2e-10);
    REQUIRE(actual.charges_.size()==expected.charges_.size());
    for(int i=0;i<actual.charges_.size();++i) REQUIRE(actual.charges_[i]==Approx(expected.charges_[i]).margin(1e-13));
}
template<class T> std::string append_serialized(PotMTP_M<T>& p) {
    std::ostringstream out;p.Save(out);return out.str();
}
}

TEMPLATE_TEST_CASE("MTP-M species append clones all species heads and preserves old and cloned predictions",
    "[mtp_m][species][append]",m_model_mtpli,m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    const bool split=GENERATE(false,true);const int long_range=GENERATE(0,1,2);
    CAPTURE(split,long_range);
    PotMTP_M<TestType> p;initialize_append(p,split,long_range);
    p.set_element_map({{0,"Si"},{1,"C"}});
    auto old_atom=mtpm_append_geometry();Neighborhoods old_nbhs(old_atom,p.cutoff);
    auto workspace=p.make_evaluation_workspace();p.getEFS(old_atom,old_nbhs,workspace);
    Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(9,-.2,.3),old_product;
    p.applyHessianVectorProduct(old_atom,old_nbhs,direction,old_product,workspace);
    Eigen::VectorXd old(p.get_NL_coeffs_count());p.nl_get_coeffs(old.data());
    const std::vector<double> shared_head(p.model_coeffs,p.model_coeffs+p.model_coeff_size);
    const std::vector<double> shared_moments(p.moment_coeffs,p.moment_coeffs+p.moment_coeff_size);
    const auto sog_weights=p.model_les.sog_weights, widths=p.model_les.sog_log_bandwidths;
    AnyPotential& api=p;REQUIRE(api.supports_species_append());
    const auto map=api.append_species({1,0},{"Ge","Al"});
    REQUIRE(p.species_count==4);REQUIRE(map.old_to_new_parameters.size()==old.size());
    Eigen::VectorXd current(p.get_NL_coeffs_count());p.nl_get_coeffs(current.data());
    for(int i=0;i<old.size();++i) REQUIRE(current[map.old_to_new_parameters[i]]==old[i]);
    REQUIRE(std::vector<double>(p.model_coeffs,p.model_coeffs+p.model_coeff_size)==shared_head);
    REQUIRE(std::vector<double>(p.moment_coeffs,p.moment_coeffs+p.moment_coeff_size)==shared_moments);
    REQUIRE(p.model_les.sog_weights==sog_weights);REQUIRE(p.model_les.sog_log_bandwidths==widths);
    REQUIRE(p.potential_tag.elements()==std::vector<std::string>{"Si","C","Ge","Al"});
    for(int property=0;property<p.n_props;++property) {
        REQUIRE(p.species_coeffs[4+property*4+2]==p.species_coeffs[4+property*4+1]);
        REQUIRE(p.species_coeffs[4+property*4+3]==p.species_coeffs[4+property*4]);
    }
    if(long_range) for(int channel=0;channel<p.les_channels();++channel) {
        REQUIRE(p.les_species[channel*4+2]==p.les_species[channel*4+1]);
        REQUIRE(p.les_species[channel*4+3]==p.les_species[channel*4]);
    }
    auto old_again=mtpm_append_geometry(),clone=mtpm_append_geometry();clone.chemmap_={3,2,3};
    Neighborhoods old_again_nbhs(old_again,p.cutoff),clone_nbhs(clone,p.cutoff);
    p.getEFS(old_again,old_again_nbhs,workspace);p.getEFS(clone,clone_nbhs,workspace);
    append_same_efs(old_again,old_atom);append_same_efs(clone,old_atom);
    Eigen::VectorXd product;p.applyHessianVectorProduct(clone,clone_nbhs,direction,product,workspace);
    REQUIRE((product-old_product).norm()<2e-10);
    if(split) {
        REQUIRE(p.p_radical_basis_pair[2*4+3]!=p.p_radical_basis_pair[1*4]);
        REQUIRE(p.p_radical_basis_pair[2*4+3]->name==p.p_radical_basis_pair[1*4]->name);
        REQUIRE(p.cutoffs(2,3)==p.cutoffs(1,0));
    }
    PotMTP_M<TestType> loaded;GPSettings settings;append_head_settings(settings);
    loaded.model.init(settings);loaded.lr_les=p.lr_les;
    std::istringstream saved(append_serialized(p));loaded.Load(saved);
    auto roundtrip=clone;loaded.getEFS(roundtrip);append_same_efs(roundtrip,clone);
    const auto state=append_serialized(p);
    REQUIRE_THROWS(api.append_species({4},{"B"}));REQUIRE_THROWS(api.append_species({-1},{"B"}));
    REQUIRE_THROWS(api.append_species({0}));REQUIRE_THROWS(api.append_species({0},{"Si"}));
    REQUIRE_THROWS(api.append_species({0},{"invalid"}));REQUIRE(append_serialized(p)==state);
    const auto noop=api.append_species({});REQUIRE(append_serialized(p)==state);
    for(int i=0;i<current.size();++i) REQUIRE(noop.old_to_new_parameters[i]==i);
}

TEMPLATE_TEST_CASE("MTP-M append remaps active packed masks and regularization without unfreezing heads",
    "[mtp_m][species][append][mask]",m_model_mtpli,m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    const int mask=GENERATE(1,2,4,8,5,10,15);
    PotMTP_M<TestType> p;initialize_append(p,false,2);
    p.req_radical=mask&1;p.req_species=mask&2;p.req_moment=mask&4;p.req_model=mask&8;
    Eigen::VectorXd before(p.get_NL_coeffs_count());p.nl_get_coeffs(before.data());
    p.reg_L1_vector.resize(before.size());p.reg_L2_vector.resize(before.size());
    for(int i=0;i<before.size();++i) {p.reg_L1_vector[i]=.01*(i+1);p.reg_L2_vector[i]=.02*(i+1);}
    const auto l1=p.reg_L1_vector,l2=p.reg_L2_vector;
    const auto map=p.append_species({1});Eigen::VectorXd after(p.get_NL_coeffs_count());p.nl_get_coeffs(after.data());
    REQUIRE(map.old_to_new_parameters.size()==before.size());
    for(int i=0;i<before.size();++i) {
        const int mapped=map.old_to_new_parameters[i];REQUIRE(after[mapped]==before[i]);
        REQUIRE(p.reg_L1_vector[mapped]==l1[i]);REQUIRE(p.reg_L2_vector[mapped]==l2[i]);
    }
    REQUIRE(p.req_radical==bool(mask&1));REQUIRE(p.req_species==bool(mask&2));
    REQUIRE(p.req_moment==bool(mask&4));REQUIRE(p.req_model==bool(mask&8));
}

TEST_CASE("MTP-M append rejects borrowed unknown and corrected layouts without mutation","[mtp_m][species][append]") {
    struct Unknown:m_model_mtpq2 {};
    PotMTP_M<Unknown> unknown;REQUIRE_FALSE(unknown.supports_species_append());
    REQUIRE_THROWS(unknown.append_species({0}));
    PotMTP_M<m_model_nn2> p;initialize_append(p,false);
    const auto state=append_serialized(p);p.coeffs_handled_external=true;
    REQUIRE_FALSE(p.supports_species_append());REQUIRE_THROWS(p.append_species({0}));p.coeffs_handled_external=false;
    p.hc_correction=true;REQUIRE_FALSE(p.supports_species_append());REQUIRE_THROWS(p.append_species({0}));p.hc_correction=false;
    REQUIRE(append_serialized(p)==state);
}
