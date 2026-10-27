#include "potentials/native_inference.h"
#include "unitest/maxvol_mtp_fixture.h"
#include <mpi.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
std::array<double,6> tensor(const Eigen::Matrix3d &w)
{
  return {w(0,0),w(1,1),w(2,2),.5*(w(0,1)+w(1,0)),
    .5*(w(0,2)+w(2,0)),.5*(w(1,2)+w(2,1))};
}
void check(double a,double b,const char *name)
{
  if (std::abs(a-b)>1e-8+2e-9*std::max(std::abs(a),std::abs(b)))
    throw std::runtime_error(std::string("Native inference vs original model mismatch: ")+name);
}
}

int main(int argc,char **argv)
{
  MPI_Init(&argc,&argv);
  int status = 0;
  try {
    if (argc != 2) throw std::runtime_error("Usage: generate_native_test_models OUTPUT_DIRECTORY");
    const auto directory = std::filesystem::absolute(argv[1]);
    auto cases = make_maxvol_mtp_models(directory);
    for (bool split : {false,true}) {
      Potentials::PotMTP model;
      load_swap_fixture(model,2,split);
      model.set_element_map({{0,"Al"},{1,"Ni"}});
      const std::string name = split ? "mtp_split" : "mtp";
      const auto path = (directory/name).string();
      model.Save(path);
      cases.push_back({name,"mtp",path,{}});
    }
    for (int mode = 0; mode < 4; ++mode) {
      const std::string names[] = {"hybrid_mtp","hybrid_zbl","scale_distance","scale_coordination"};
      const std::string type = mode < 2 ? "hybrid" : "hybrid_scale";
      GPSettings settings;
      settings.Modify("pottype",type); settings.Modify("npot","2");
      settings.Modify("pot1:pottype","mtp"); settings.Modify("pot1:potfile","mtp");
      settings.Modify("pot1:train","false");
      settings.Modify("pot2:pottype",mode == 1 ? "zbl" : "mtp");
      if (mode == 1) {
        settings.Modify("pot2:z","2 3"); settings.Modify("pot2:r0","1.5"); settings.Modify("pot2:r1","3.8");
      } else {
        settings.Modify("pot2:potfile","mtp_split"); settings.Modify("pot2:train","false");
      }
      if (mode >= 2) {
        settings.Modify("feature",mode == 2 ? "min_distance" : "smooth_coordination");
        settings.Modify("feature_r0","0.5"); settings.Modify("feature_r1","6.5");
        settings.Modify("boundary1",mode == 2 ? "1.5 3.5" : "0.2 1.8");
      }
      const auto path = directory/(names[mode]+".manifest");
      std::ofstream manifest(path);
      for (const auto &[key,value] : settings.settings) manifest << key << ' ' << value << '\n';
      manifest.close();
      cases.push_back({names[mode],type,path.string(),{}});
    }
    {
      Potentials::PotMTP_LES model((directory/"sog").string());
      model.sog_channels = 3;
      model.train_init();
      for (int c = 0; c < 3; ++c) {
        for (int s = 0; s < model.species_count; ++s)
          model.les_species_coeffs[c*model.species_count+s] = .13*(c+1)-.07*s;
        for (int a = 0; a < model.alpha_scalar_moments; ++a)
          model.les_moment_coeffs[c*model.alpha_scalar_moments+a] = .02*(c+1)/(a+1);
      }
      const auto path = (directory/"sog3").string();
      std::ofstream stream(path); model.Save(stream); stream.close();
      cases.push_back({"sog3","mtp_les",path,{}});
    }
    for (int layers : {1,2}) for (int activation = 0; activation < 3; ++activation) {
      const std::string name = "nn"+std::to_string(layers)+"_activation"+std::to_string(activation);
      Potentials::PotMTPNN1 model((directory/("nn"+std::to_string(layers))).string());
      auto configure = [&](auto &network) {
        network.acfunc = static_cast<IMRMLPNN::ActivationFunction>(activation);
        for (int i = 0; i < network.n_input; ++i) {
          network.norm_mean[i] = .2-.13*i;
          network.norm_irstd[i] = .7+.23*i;
        }
      };
      if (layers == 1) configure(*model.p_nn); else configure(*model.p_nn2);
      model.species_coeffs = {-.23,.17};
      const auto path = (directory/name).string(); model.Save(path);
      cases.push_back({name,"mtpnn1",path,{}});
    }
    for (const std::string head : {"mtp_m_n1","mtp_m_n2"})
      for (const std::string activation : {"RELU","SIGMOID","LINEAR"}) {
        if (head == "mtp_m_n2" && activation == "LINEAR") continue;
        const std::string name = head+"_"+activation;
        GPSettings settings;
        settings.Modify("pottype","mtp_m"); settings.Modify("extmodel",head);
        settings.Modify("potfile",(directory/"template.mtp").string());
        settings.Modify("activation_function",activation);
        settings.Modify("n_input","2"); settings.Modify("n_neuron","3"); settings.Modify("n_neuron2","2");
        Potentials::potential_wrapper wrapper(settings);
        wrapper.pot->train_init();
        std::vector<double> coefficients(wrapper.pot->get_NL_coeffs_count());
        for (std::size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = .06+.01*std::cos(i);
        wrapper.pot->nl_set_coeffs(coefficients.data());
        const auto path = (directory/name).string(); wrapper.pot->Save(path);
        settings.Modify("potfile",path);
        cases.push_back({name,"mtp_m",path,settings});
      }
    for (const std::string head : {"mtp_m_fs","mtp_m_li","mtp_m_n1","mtp_m_n2"})
      for (bool sog : {false,true}) {
        const std::string name = head+(sog ? "_sog" : "_les");
        std::ifstream base(directory/"template.mtp");
        std::ostringstream contents; contents << base.rdbuf();
        const auto source_path = directory/(name+".template");
        std::ofstream source(source_path);
        source << contents.str() << "les_species_coeffs = {0.2, -0.15}\n"
               << "les_moment_coeffs = {0.05, -0.02, 0.01}\n";
        if (sog) {
          Potentials::SOGModel definition;
          definition.weights = {1.2,-.3}; definition.log_bandwidths = {-.4,.2};
          definition.write(source);
        }
        source.close();
        GPSettings settings;
        settings.Modify("pottype","mtp_m"); settings.Modify("extmodel",head);
        settings.Modify("potfile",source_path.string()); settings.Modify("lr_les","true");
        settings.Modify("n_input","2"); settings.Modify("n_neuron","3"); settings.Modify("n_neuron2","2");
        Potentials::potential_wrapper wrapper(settings);
        wrapper.pot->train_init();
        std::vector<double> coefficients(wrapper.pot->get_NL_coeffs_count());
        for (std::size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = .06+.01*std::cos(i);
        wrapper.pot->nl_set_coeffs(coefficients.data());
        const auto path = (directory/name).string(); wrapper.pot->Save(path);
        settings.Modify("potfile",path);
        cases.push_back({name,"mtp_m",path,settings});
      }
    {
      // A manufactured bound dimer: V(r)=0.1*(r-2.5)^2*(r-5)^2.
      // Chebyshev coordinates use r=2.25*x+2.75 on [0.5,5].
      std::ofstream stable(directory/"stable-dimer.mtp");
      stable << "MTP\nversion = 1.1.0\npotential_name = StableDimer\nscaling = 1\n"
        "species_count = 1\npotential_tag =\nradial_basis_type = RBChebyshev\n"
        "min_dist = 0.5\nmax_dist = 5\nradial_basis_size = 3\nradial_funcs_count = 1\n"
        "radial_coeffs\n0-0\n{2.59375, 1.125, 2.53125}\n"
        "alpha_moments_count = 1\nalpha_index_basic_count = 1\n"
        "alpha_index_basic = {{0, 0, 0, 0}}\nalpha_index_times_count = 0\nalpha_index_times = {}\n"
        "alpha_scalar_moments = 1\nalpha_moment_mapping = {0}\nspecies_coeffs = {0}\nmoment_coeffs = {0.05}\n";
    }
    std::ofstream manifest(directory/"cases.tsv");
    manifest << "name\tpottype\tmodel\tsettings\tgeometry\treference\tcharges\tchannels\n";
    for (const auto &test : cases) {
      const auto settings_file = directory/(test.name+".ini");
      std::ofstream settings(settings_file);
      for (const auto &[key,value] : test.settings.settings)
        if (key != "potfile" && key != "pottype" && !value.empty()) settings << "pot:" << key << ' ' << value << '\n';
      settings.close();
      Potentials::NativeInferenceOptions options;
      options.pottype = test.pottype; options.settings_file = settings_file;
      {
        Potentials::NativeInference updating(test.path,options);
        auto &potential = updating.potential();
        potential.train_init();
        std::vector<double> coefficients(potential.get_NL_coeffs_count());
        if (!coefficients.empty()) {
          potential.nl_get_coeffs(coefficients.data());
          for (std::size_t i = 0; i < coefficients.size(); ++i)
            coefficients[i] += 1e-5*std::sin(.31*(i+1));
          potential.nl_set_coeffs(coefficients.data());
        }
        potential.Save(test.path);
      }
      Potentials::NativeInference evaluator(test.path,options);
      const bool gpm=std::getenv("IMRMLP_NATIVE_GPM")!=nullptr;
      const std::string deployed_path=gpm?test.path+".gpm":test.path;
      if(gpm)evaluator.potential().Save(deployed_path);
      std::vector<std::string> geometries{"orthogonal","triclinic"};
      if (test.name == "scale_distance") geometries.push_back("tie");
      if (std::getenv("IMRMLP_NATIVE_FOUR_RANK") &&
          (test.name == "les" || test.name == "sog3" ||
           test.name == "mtp_m_n2_sog")) geometries.push_back("four_rank");
      for (const std::string &geometry : geometries) {
        auto configuration = maxvol_mtp_atoms();
        if (geometry == "triclinic") {
          configuration.cell_(1,0) = 1.1;
          configuration.cell_(2,0) = -.7;
          configuration.cell_(2,1) = .8;
        }
        if (geometry == "cluster") configuration.pbc_ = {false,false,false};
        if (geometry == "tie") configuration.pos_ = {{6.,6.,6.},{8.,6.,6.},{4.,6.,6.}};
        if (geometry == "four_rank") {
          configuration.pos_ = {{1.5,2.,2.},{4.5,2.6,2.5},{7.5,2.8,2.9},{10.5,3.1,3.3}};
          configuration.chemmap_ = {1,0,1,0};
        }
        const auto result = evaluator.evaluate(configuration);
        auto reference = configuration;
        evaluator.potential().getEFS(reference);
        check(result.energy,reference.energy_,"energy");
        for (int i = 0; i < reference.size(); ++i)
          for (int k = 0; k < 3; ++k) check(result.forces(i,k),reference.forces_[i][k],"force");
        const auto w = tensor(result.virial), original = tensor(reference.stress_);
        for (int k = 0; k < 6; ++k) {
          check(w[k],original[k],"virial");
          check(result.site_virial.col(k).sum(),w[k],"site virial sum");
        }
        check(result.site_energy.sum(),result.energy,"site energy sum");
        const auto output = directory/(test.name+"-"+geometry+".reference");
        std::ofstream out(output); out << std::setprecision(17) << result.energy;
        for (double value : w) out << ' ' << value;
        out << '\n';
        for (int i = 0; i < reference.size(); ++i) {
          out << i+1;
          for (int k = 0; k < 3; ++k) out << ' ' << result.forces(i,k);
          out << ' ' << result.site_energy(i);
          for (int k = 0; k < 6; ++k) out << ' ' << result.site_virial(i,k);
          if (evaluator.has_charges()) out << ' ' << result.charges(i);
          for (int c = 0; c < evaluator.latent_channels(); ++c) out << ' ' << result.latents(i,c);
          out << '\n';
        }
        manifest << test.name << '\t' << (gpm?"auto":test.pottype) << '\t' << deployed_path << '\t' << (gpm?"":settings_file.string())
          << '\t' << geometry << '\t' << output.string() << '\t' << evaluator.has_charges()
          << '\t' << evaluator.latent_channels() << '\n';
      }
    }
    if(std::getenv("IMRMLP_NATIVE_GPM"))for(const auto &entry:std::filesystem::directory_iterator(directory))
      if(entry.is_regular_file()&&entry.path().extension()!=".gpm"&&entry.path().extension()!=".reference"&&entry.path().filename()!="cases.tsv")
        std::filesystem::remove(entry.path());
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; status = 1; }
  MPI_Finalize();
  return status;
}
