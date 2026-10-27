#pragma once
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include <fstream>
#include <sstream>
namespace LESCurvature {
using namespace Potentials;
Atoms geometry(bool small = false) {
    Atoms atom;
    atom.cell_ << 10.,0,0,.8,10.5,0,-.3,.6,11.;
    atom.pos_ = {{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    atom.chemmap_ = {0,1,0,1};
    if (small) {
        atom.cell_ << 3.8,.3,.1,.1,4.1,.2,.2,.1,4.3;
        atom.pos_ = {{.3,.5,.4},{1.65,.8,.7},{2.2,2.4,1.2}};
        atom.chemmap_ = {0,1,0};
    }
    return atom;
}
void configure_engine(LES& engine, bool sog, bool self = true) {
    engine.kernel = sog ? LES::Kernel::SOG : LES::Kernel::Coulomb;
    engine.dl = 2.;
    engine.remove_self_interaction = self;
    if (sog) {
        engine.sog_weights = {4.2,-1.3};
        engine.sog_log_bandwidths = {-.4,.2};
    }
}
void initialize(PotMTP_LES& model, bool sog) {
    model.les_kernel = sog ? "sog" : "coulomb";
    model.sog_channels = sog ? 2 : 1;
    model.sog_terms = 2;
    model.train_init();
    for (int c = 0; c < model.channels(); ++c) {
        model.les_species_coeffs[2*c] = .23+.11*c;
        model.les_species_coeffs[2*c+1] = -.17+.07*c;
        for (int a = 0; a < model.alpha_scalar_moments; ++a)
            model.les_moment_coeffs[c*model.alpha_scalar_moments+a] = a < 4 ? .12*std::sin(a+c+1.) : 0.;
    }
    if (sog) {
        model.model_les.sog_weights = {4.2,-1.3,3.1,-.8};
        model.model_les.sog_log_bandwidths = {-.4,.2,-.2,.1};
    }
}
template<class Model> void initialize(PotMTP_M<Model>& model, bool sog) {
    GPSettings settings;
    settings.Modify("n_input","2"); settings.Modify("n_neuron","3");
    model.model.init(settings);
    std::ifstream file("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE(file.good());
    std::string source((std::istreambuf_iterator<char>(file)),{});
    source.resize(source.find("species_coeffs"));
    PotMTP reference("unitest/test-data/portable-mtp/les.mtp");
    const int count = reference.alpha_scalar_moments, channels = sog ? 2 : 1;
    source += "les_species_coeffs = {";
    for (int c = 0; c < channels; ++c) {
        if (c) source += ", ";
        source += std::to_string(.23+.11*c)+", "+std::to_string(-.17+.07*c);
    }
    source += "}\nles_moment_coeffs = {";
    for (int c = 0; c < channels; ++c)
        for (int a = 0; a < count; ++a) {
            if (c || a) source += ", ";
            source += std::to_string(a < 4 ? .12*std::sin(a+c+1.) : 0.);
        }
    source += "}\n";
    if (sog) {
        SOGModel definition;
        definition.channels = 2;
        definition.weights = {4.2,-1.3,3.1,-.8};
        definition.log_bandwidths = {-.4,.2,-.2,.1};
        std::ostringstream metadata; definition.write(metadata); source += metadata.str();
    } else model.lr_les = true;
    std::istringstream input(source); model.Load(input);
    for (int i = 0; i < model.species_coeff_size; ++i) model.species_coeffs[i] = .04*std::sin(i+1.);
    for (int i = 0; i < model.moment_coeff_size; ++i) model.moment_coeffs[i] = .04*std::sin(i+1.);
    for (int i = 0; i < model.model_coeff_size; ++i) model.model_coeffs[i] = .07*std::sin(i+1.);
    model.train_init();
    // This fixture intentionally omitted SR coefficient sections while loading
    // the graph, then supplied every species/moment/network value above.
    model.empty=false;
    model.set_inited();
}
Eigen::VectorXd force_fd(AnyPotential& model, const Atoms& atom,
    const Eigen::VectorXd& direction, double step, double cutoff) {
    auto plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step*direction.segment<3>(3*i).transpose();
        minus.pos_[i] -= step*direction.segment<3>(3*i).transpose();
    }
    Neighborhoods np(plus,cutoff),nm(minus,cutoff);
    model.getEFS(plus,np); model.getEFS(minus,nm);
    Eigen::VectorXd result(direction.size());
    for (int i = 0; i < atom.size(); ++i)
        result.segment<3>(3*i) = -(plus.forces_[i]-minus.forces_[i])/(2*step);
    return result;
}
}
