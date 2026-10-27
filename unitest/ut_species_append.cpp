#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include "train/train_nonlinear_potential.h"
#include <sstream>

using namespace Potentials;
namespace {
std::string append_fixture(bool split) {
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

Atoms append_geometry() {
    Atoms a;a.cell_=Eigen::Matrix3d::Identity()*12.;
    a.pos_={{1.,1.,1.},{2.1,1.3,1.2},{1.4,2.4,1.6}};
    a.chemmap_={0,1,0};return a;
}
void same_efs(const Atoms& a,const Atoms& b) {
    REQUIRE(a.energy_==Catch::Approx(b.energy_).margin(1e-12));
    REQUIRE((a.stress_-b.stress_).norm()<1e-11);
    REQUIRE(a.forces_.size()==b.forces_.size());
    for(size_t i=0;i<a.forces_.size();++i) REQUIRE((a.forces_[i]-b.forces_[i]).norm()<1e-11);
}
std::string serialized(PotMTP& p) {std::ostringstream out;p.Save(out);return out.str();}
}

TEST_CASE("ordinary MTP append clones explicit species and preserves old layout mapping","[mtp][species][append]") {
    const bool split=GENERATE(false,true);
    PotMTP potential;std::istringstream input(append_fixture(split));potential.Load(input);
    potential.set_element_map({{0,"Si"},{1,"C"}});
    auto old_geometry=append_geometry();potential.getEFS(old_geometry);
    std::vector<double> old(potential.get_NL_coeffs_count());potential.nl_get_coeffs(old.data());
    AnyPotential& api=potential;REQUIRE(api.supports_species_append());
    const auto result=api.append_species({1,0},{"Ge","Al"});
    REQUIRE(api.species_count==4);REQUIRE(result.old_to_new_parameters.size()==old.size());
    REQUIRE(potential.potential_tag.elements()==std::vector<std::string>{"Si","C","Ge","Al"});
    std::vector<double> current(potential.get_NL_coeffs_count());potential.nl_get_coeffs(current.data());
    for(size_t i=0;i<old.size();++i) REQUIRE(current.at(result.old_to_new_parameters[i])==old[i]);
    auto predicted=append_geometry();api.getEFS(predicted);same_efs(predicted,old_geometry);
    auto clone=append_geometry();clone.chemmap_={3,2,3};api.getEFS(clone);same_efs(clone,old_geometry);
    REQUIRE(potential.species_coeffs[2]==potential.species_coeffs[1]);
    REQUIRE(potential.species_coeffs[3]==potential.species_coeffs[0]);
    if(split) {
        REQUIRE(potential.p_radical_basis_pair[2*4+3]!=potential.p_radical_basis_pair[1*4]);
        REQUIRE(potential.p_radical_basis_pair[2*4+3]->name==potential.p_radical_basis_pair[1*4]->name);
        REQUIRE(potential.cutoffs(2,3)==potential.cutoffs(1,0));
    }
    std::istringstream saved(serialized(potential));PotMTP loaded;loaded.Load(saved);
    REQUIRE(loaded.species_count==4);REQUIRE(loaded.potential_tag.elements()==potential.potential_tag.elements());
    auto roundtrip=clone;loaded.getEFS(roundtrip);same_efs(roundtrip,clone);
    const auto state=serialized(potential);
    REQUIRE_THROWS(api.append_species({4},{"B"}));
    REQUIRE_THROWS(api.append_species({-1},{"B"}));
    REQUIRE_THROWS(api.append_species({0}));
    REQUIRE_THROWS(api.append_species({0},{"Si"}));
    REQUIRE_THROWS(api.append_species({0},{"invalid"}));
    REQUIRE(serialized(potential)==state);
    const auto no_op=api.append_species({});REQUIRE(serialized(potential)==state);
    for(size_t i=0;i<current.size();++i) REQUIRE(no_op.old_to_new_parameters[i]==int(i));
}

TEST_CASE("species append invalidates initialized trainer parameter buffers","[mtp][species][append][training]") {
    PotMTP potential;std::istringstream input(append_fixture(false));potential.Load(input);
    auto reference=append_geometry();potential.getEFS(reference);reference.have_energy=true;
    std::vector<Atoms> data{reference};std::vector<Neighborhoods> nbhs{Neighborhoods(reference,potential.cutoff)};
    GPSettings settings;settings.Modify("save_to","");settings.Modify("weight_force","0");settings.Modify("weight_stress","0");
    TrainNonlinear trainer(settings);trainer.init(data,nbhs,&potential);
    potential.append_species({1});const auto state=serialized(potential);
    double loss=0.;Eigen::VectorXd gradient=Eigen::VectorXd::Zero(1);
    REQUIRE_THROWS_WITH(trainer.AccumulateLoss(loss),Catch::Matchers::ContainsSubstring("reinitialize"));
    REQUIRE_THROWS_WITH(trainer.AccumulateGard(loss,gradient),Catch::Matchers::ContainsSubstring("reinitialize"));
    REQUIRE_THROWS_WITH(trainer.TrainNonLinearBFGS(0),Catch::Matchers::ContainsSubstring("reinitialize"));
    REQUIRE(serialized(potential)==state);
    trainer.init(data,nbhs,&potential);trainer.AccumulateLoss(loss);REQUIRE(std::isfinite(loss));
}

TEST_CASE("species append rejects unsupported derived and borrowed MTP models","[mtp][species][append]") {
    class DerivedMTP : public PotMTP {};
    DerivedMTP derived;std::istringstream input(append_fixture(false));derived.Load(input);
    AnyPotential& api=derived;REQUIRE_FALSE(api.supports_species_append());
    const auto state=serialized(derived);REQUIRE_THROWS(api.append_species({0}));
    REQUIRE(serialized(derived)==state);
    PotMTP borrowed;std::istringstream second(append_fixture(false));borrowed.Load(second);
    borrowed.coeffs_handled_external=true;
    REQUIRE_FALSE(borrowed.supports_species_append());REQUIRE_THROWS(borrowed.append_species({0}));
    borrowed.coeffs_handled_external=false;
}

TEST_CASE("ordinary species append remaps raw regularization and invalidates unknown trainer layouts", "[mtp][species][append][regularization]") {
    const bool empty=GENERATE(false,true);
    PotMTP potential;std::istringstream input(append_fixture(false));potential.Load(input);
    potential.train_init();
    const int old_radial=potential.radial_coeff_size,old_count=potential.species_count;
    const int old_parameters=potential.reg_coeff_size,stride=potential.radial_func_count*potential.rb_size;
    potential.reg_L1_vector.resize(old_parameters);potential.reg_L2_vector.resize(old_parameters);
    for(int i=0;i<old_parameters;++i) {
        potential.reg_L1_vector[i]=.01*(i+1);potential.reg_L2_vector[i]=.02*(i+1);
    }
    const auto l1=potential.reg_L1_vector,l2=potential.reg_L2_vector;
    if(empty) {potential.reg_L1_vector.clear();potential.reg_L2_vector.clear();}
    const auto appended=potential.append_species({1});
    REQUIRE(potential.reg_L1_vector.size()==size_t(potential.reg_coeff_size));
    REQUIRE(potential.reg_L2_vector.size()==size_t(potential.reg_coeff_size));
    if(empty) {
        REQUIRE(potential.reg_L1_vector==std::vector<double>(potential.reg_coeff_size,0.));
        REQUIRE(potential.reg_L2_vector==std::vector<double>(potential.reg_coeff_size,0.));
    } else {
        for(int i=0;i<old_parameters;++i) {
            REQUIRE(potential.reg_L1_vector[appended.old_to_new_parameters[i]]==l1[i]);
            REQUIRE(potential.reg_L2_vector[appended.old_to_new_parameters[i]]==l2[i]);
        }
        const std::vector<int> source={0,1,1};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int a=0;a<stride;++a) {
            const int dst=(i*3+j)*stride+a,src=(source[i]*old_count+source[j])*stride+a;
            REQUIRE(potential.reg_L1_vector[dst]==l1[src]);REQUIRE(potential.reg_L2_vector[dst]==l2[src]);
        }
        REQUIRE(potential.reg_L1_vector[potential.radial_coeff_size+2]==l1[old_radial+1]);
        REQUIRE(potential.reg_L2_vector[potential.radial_coeff_size+2]==l2[old_radial+1]);
    }
    // No trainer has been rebuilt yet: the potential's penalty API must still
    // have safely indexable, full-size regularization storage.
    Eigen::VectorXd gradient=Eigen::VectorXd::Zero(potential.get_NL_coeffs_count());double loss=0.;
    REQUIRE_NOTHROW(potential.AddPenaltyGard(0.,loss,gradient.data()));
    REQUIRE(std::isfinite(loss));REQUIRE(gradient.allFinite());
}
