// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "utils/lower_energy_envelope.h"
#include <algorithm>
#include <array>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <cmath>
#include <stdexcept>

namespace {
void validate_score(bool enabled, double scale, double reference, double power, const char *label) {
    if (enabled && (!std::isfinite(scale) || scale <= 0.0 ||
                    !std::isfinite(reference) || !std::isfinite(power)))
        throw std::invalid_argument(std::string(label) + " fitness requires a finite positive scale and finite reference/power");
}
double score(double statistic, double scale, double reference, double power) {
    if (!std::isfinite(statistic))
        throw std::invalid_argument("Fitness statistic must be finite");
    const double factor = statistic > reference
        ? scale / (scale + std::pow(statistic - reference, power)) : 1.0;
    if (!std::isfinite(factor) || factor < 0.0 || factor > 1.0)
        throw std::runtime_error("Fitness factor must be finite and in [0,1]");
    return factor;
}
constexpr std::array<const char *, 3> hull_diagnostics = {
    "hull_energy_per_atom", "energy_relative_to_hull", "energy_above_hull"};

void clear_hull_diagnostics(Atoms &atom) {
    for (const char *key : hull_diagnostics) {
        atom.global_attribs_d_.erase(key);
        atom.global_attribs_i_.erase(key);
        atom.features_.erase(key);
    }
}

void set_hull_diagnostic(Atoms &atom, const char *key, double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Hull diagnostics must be finite");
    atom.global_attribs_d_[key] = value;
    // The command's established mlpcfg output carries generic scalars as
    // Feature records. Keep the typed value for in-memory consumers too.
    std::ostringstream text;
    text << std::setprecision(17) << value;
    atom.features_[key] = text.str();
}

double energy_per_atom(const Atoms &atom) {
    if (!atom.has_energy() || atom.size() <= 0 || !std::isfinite(atom.energy_))
        throw std::invalid_argument("Hull scoring requires finite energy and a nonempty configuration");
    return atom.energy_ / atom.size();
}

std::vector<int> composition_key(const Atoms &atom, const std::vector<int> &species) {
    if (atom.size() <= 0 || atom.chemmap_.size() != static_cast<std::size_t>(atom.size()))
        throw std::invalid_argument("Hull composition requires a nonempty configuration with species for every atom");
    std::vector<int> counts(species.size(), 0);
    for (int type : atom.chemmap_) {
        const auto it = std::lower_bound(species.begin(), species.end(), type);
        if (it == species.end() || *it != type)
            throw std::invalid_argument("Unknown hull composition species");
        ++counts[it - species.begin()];
    }
    int divisor = 0;
    for (int n : counts) divisor = std::gcd(divisor, n);
    for (int &n : counts) n /= divisor;
    return counts;
}

LowerEnergyEnvelope::Coordinates composition_coordinates(const std::vector<int> &key) {
    const double total = std::accumulate(key.begin(), key.end(), 0.0);
    LowerEnergyEnvelope::Coordinates result;
    for (int n : key) result.push_back(n / total);
    return result;
}

LowerEnergyEnvelope::Coordinates property_coordinates(const Atoms &atom, const std::vector<std::string> &keys) {
    LowerEnergyEnvelope::Coordinates result;
    for (const auto &key : keys) {
        const auto d = atom.global_attribs_d_.find(key);
        const auto i = atom.global_attribs_i_.find(key);
        const bool have_d = d != atom.global_attribs_d_.end();
        const bool have_i = i != atom.global_attribs_i_.end();
        if (have_d && have_i)
            throw std::invalid_argument("Hull coordinate '" + key + "' is ambiguous between global double and integer scalars");
        if (!have_d && !have_i)
            throw std::invalid_argument("Hull coordinate '" + key + "' requires an existing global double or integer scalar");
        const double value = have_d ? d->second : static_cast<double>(i->second);
        if (!std::isfinite(value))
            throw std::invalid_argument("Hull coordinate '" + key + "' must be finite");
        result.push_back(value);
    }
    return result;
}
}

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) try
{
    GPIO iobj;
    GPSettings settings;
    string settingfile = "input.inp";
    std::vector<Atoms> atoms;
    std::vector<Atoms> resatoms;
    string informat = "";
    string infile = "input.cfg";
    string outfile = "output-fitness.cfg";
    string outformat = "mlpcfg";
    string indexfile = "index.map"; // may contains reference energy
    bool sel_by_eng = true;
    bool sel_by_force = false;
    bool sel_by_stress = false;
    string energy_mode = "formation";
    std::vector<string> hull_coordinates;
    string hull_reference, hull_reference_format;
    double hull_ref = 0.0, hull_scale = 0.1, hull_power = 1.0;
    double force_dbl = 1.0;
    double force_pow = 1.0;
    double force_ref = 0.0;
    double stress_ref = 0.0;
    double stress_dbl = 100;
    double stress_pow = 1.0;
    double eng_ref = -50; 
    double fit_dbl = 10;
    double fit_pow = 1.0;
    double fitness_threshold = -1;
    bool divide_size = true;
    CMDOption::CommandLineOptions cmdopts;
    cmdopts.set_helponexit(false);
    cmdopts.addop(indexfile, "index", "id", "index file", false, 1);
    cmdopts.addop(informat, "input-format", "if", "format, use XDATCAR or lammpstrj or mlpcfg", false, 1);
    cmdopts.addop(energy_mode, "energy-mode", "", "formation, composition-hull, or property-hull", false, 1);
    cmdopts.addop(hull_coordinates, "hull-coordinates", "", "global scalar keys for the mathematical property lower envelope (convex mixing, not general thermodynamic stability)", false, -1);
    cmdopts.addop(hull_reference, "hull-reference", "", "fixed hull reference trajectory; default is all energy-bearing input configurations", false, 1);
    cmdopts.addop(hull_reference_format, "hull-reference-format", "", "reference format; default is guessed from the reference filename", false, 1);
    cmdopts.addop(hull_ref, "hullref", "", "energy-above-hull reference in eV/atom", false, 1);
    cmdopts.addop(hull_scale, "hullevl", "", "energy-above-hull fitness scale (default 0.1)", false, 1);
    cmdopts.addop(hull_power, "hullpow", "", "energy-above-hull fitness power", false, 1);
    cmdopts.addop(sel_by_eng, "select_by_energy", "ue", "use energy", false, 1);
    cmdopts.addop(sel_by_force, "select_by_force", "uf", "use force", false, 1);
    cmdopts.addop(sel_by_stress, "select_by_stress", "us", "use stress", false, 1);
    cmdopts.addop(divide_size, "divide_by_size", "div", "normalize formation energy by atom count", false, 1);
    cmdopts.addop(eng_ref, "engref", "er", "reference energy", false, 1);
    cmdopts.addop(fit_dbl, "engevl", "ev", "whose weight is 0.5", false, 1);
    cmdopts.addop(fit_pow, "engpow", "ep", "power", false, 1);
    cmdopts.addop(force_ref, "forceref", "fr", "reference squared force", false, 1);
    cmdopts.addop(force_dbl, "forceevl", "fv", "whose weight is 0.5", false, 1);
    cmdopts.addop(force_pow, "forcepow", "fp", "power", false, 1);
    cmdopts.addop(stress_ref, "stressref", "sr", "reference stress", false, 1);
    cmdopts.addop(stress_dbl, "stressevl", "sv", "whose weight is 0.5", false, 1);
    cmdopts.addop(stress_pow, "stresspow", "sp", "power", false, 1);
    cmdopts.addop(settingfile, "settings", "", "settings", false, 1);
    cmdopts.addop(fitness_threshold, "threshold", "t", "select", false, 1);
    cmdopts.addop(iobj.absolute_index_read, "absolute_index_read", "abr", "whether to use atomic numbers when reading", false, 0);
    cmdopts.addop(iobj.absolute_index_write, "absolute_index_write", "abw", "whether to use atomic numbers when writing instead of 1, 2, 3, ...", false, 0);
    cmdopts.addop(infile, "trainset", "in", "load trajectory from", true, 1);
    cmdopts.addop(outfile, "output", "out", "output to file", true, 1);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        return 0;
    }
    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    if (sel_by_eng && energy_mode != "formation" && energy_mode != "composition-hull" && energy_mode != "property-hull")
        throw std::invalid_argument("Unknown energy-mode: " + energy_mode);
    const bool use_hull = sel_by_eng && energy_mode != "formation";
    const bool use_composition = energy_mode == "composition-hull";
    validate_score(sel_by_eng && !use_hull, fit_dbl, eng_ref, fit_pow, "Energy");
    validate_score(use_hull, hull_scale, hull_ref, hull_power, "Hull");
    if (use_hull && !divide_size)
        throw std::invalid_argument("Hull scoring requires divide_by_size=true (raw energy in eV/atom)");
    if (use_hull && !use_composition) {
        if (hull_coordinates.empty()) throw std::invalid_argument("property-hull requires hull-coordinates");
        const std::set<string> unique(hull_coordinates.begin(), hull_coordinates.end());
        if (unique.size() != hull_coordinates.size()) throw std::invalid_argument("Hull coordinate keys must be unique");
    }
    validate_score(sel_by_force, force_dbl, force_ref, force_pow, "Force");
    validate_score(sel_by_stress, stress_dbl, stress_ref, stress_pow, "Stress");
    if (!std::isfinite(fitness_threshold))
        throw std::invalid_argument("Fitness threshold must be finite");
    if (!iobj.readmap(indexfile)) {
        cerr << "Error: failed to read index file " << indexfile << '\n';
        return 1;
    }
    // read atoms
    Message("Reading trajectory file " + infile);
    if (!iobj.read(atoms, infile, informat) || atoms.empty()) {
        cerr << "Error: failed to read nonempty trajectory " << infile << '\n';
        return 1;
    }
    std::vector<Atoms> reference_atoms;
    std::vector<int> species;
    std::vector<LowerEnergyEnvelope::Coordinates> query_coordinates(atoms.size());
    std::vector<std::vector<int>> query_compositions(atoms.size());
    std::unique_ptr<LowerEnergyEnvelope> envelope;
    if (use_hull) {
        const auto *references = &atoms;
        if (!hull_reference.empty()) {
            if (!iobj.read(reference_atoms, hull_reference, hull_reference_format) || reference_atoms.empty())
                throw std::invalid_argument("Failed to read nonempty hull reference " + hull_reference);
            references = &reference_atoms;
        }
        std::set<int> species_set;
        if (use_composition) {
            for (const auto *set : std::array<const std::vector<Atoms> *, 2>{&atoms, references})
                for (const auto &atom : *set)
                    species_set.insert(atom.chemmap_.begin(), atom.chemmap_.end());
            species.assign(species_set.begin(), species_set.end());
        }
        std::vector<LowerEnergyEnvelope::Coordinates> coordinates;
        std::vector<double> energies;
        for (std::size_t j = 0; j < references->size(); ++j) {
            const auto &atom = (*references)[j];
            if (hull_reference.empty() && !atom.has_energy()) continue;
            try {
                energies.push_back(energy_per_atom(atom));
                coordinates.push_back(use_composition ? composition_coordinates(composition_key(atom, species))
                                                     : property_coordinates(atom, hull_coordinates));
            } catch (const std::exception &error) {
                throw std::invalid_argument("Hull reference configuration " + std::to_string(j + 1) + ": " + error.what());
            }
        }
        // If all query energies are absent there is no energy contribution.
        if (!coordinates.empty()) envelope = std::make_unique<LowerEnergyEnvelope>(coordinates, energies);
        for (std::size_t j = 0; j < atoms.size(); ++j) {
            if (!atoms[j].has_energy()) continue;
            try {
                if (use_composition) {
                    query_compositions[j] = composition_key(atoms[j], species);
                    query_coordinates[j] = composition_coordinates(query_compositions[j]);
                } else {
                    query_coordinates[j] = property_coordinates(atoms[j], hull_coordinates);
                }
            } catch (const std::exception &error) {
                throw std::invalid_argument("Hull query configuration " + std::to_string(j + 1) + ": " + error.what());
            }
        }
    }
    std::map<std::vector<int>, double> composition_cache;
    // init formation
    std::vector<double> formation_energies(atoms.size(), 0.0);
    for (int i = 0; i < atoms.size(); ++i)
    {
        if (use_hull) clear_hull_diagnostics(atoms[i]);
        if (sel_by_eng && !use_hull && atoms[i].has_energy())
        {
            formation_energies[i] = atoms[i].energy_;
            for (int j = 0; j < atoms[i].size(); ++j)
                formation_energies[i] -= iobj.atomradii[atoms[i].chemmap_[j]];
            if (divide_size)
                formation_energies[i] /= atoms[i].size();
        }
        for (const auto &[enabled, available, label] : {
                 std::tuple{sel_by_eng, atoms[i].has_energy(), "energy"},
                 std::tuple{sel_by_force, atoms[i].has_forces(), "forces"},
                 std::tuple{sel_by_stress, atoms[i].has_stresses(), "stress"}})
            if (enabled && !available)
                Warning("Configuration " + to_string(i + 1) + " has no " + label +
                        "; skipping this fitness contribution (neutral factor 1)");
        atoms[i].fitness = 1.0;
    }
    // calculate fitness
    {
        for (int i = 0; i < atoms.size(); ++i)
        {
            if (sel_by_eng && atoms[i].has_energy())
            {
                if (!use_hull) {
                    atoms[i].fitness *= score(formation_energies[i], fit_dbl, eng_ref, fit_pow);
                } else {
                    try {
                        const double energy = energy_per_atom(atoms[i]);
                        double hull_energy;
                        if (use_composition) {
                            const auto &key = query_compositions[i];
                            auto it = composition_cache.find(key);
                            if (it == composition_cache.end())
                                it = composition_cache.emplace(key, envelope->energy_at(query_coordinates[i])).first;
                            hull_energy = it->second;
                        } else {
                            hull_energy = envelope->energy_at(query_coordinates[i]);
                        }
                        double delta = energy - hull_energy;
                        if (!std::isfinite(delta))
                            throw std::invalid_argument("Energy relative to hull must be finite");
                        if (hull_reference.empty() && delta < 0) {
                            const double tolerance = 1e-8 * std::max({1.0, std::fabs(energy), std::fabs(hull_energy)});
                            if (delta < -tolerance)
                                throw std::runtime_error("Self-reference lower envelope exceeds the query energy");
                            delta = 0;
                        }
                        const double excess = std::max(0.0, delta);
                        set_hull_diagnostic(atoms[i], hull_diagnostics[0], hull_energy);
                        set_hull_diagnostic(atoms[i], hull_diagnostics[1], delta);
                        set_hull_diagnostic(atoms[i], hull_diagnostics[2], excess);
                        atoms[i].fitness *= score(excess, hull_scale, hull_ref, hull_power);
                    } catch (const std::exception &error) {
                        throw std::invalid_argument("Hull query configuration " + std::to_string(i + 1) + ": " + error.what());
                    }
                }
            }
            if (sel_by_force && atoms[i].has_forces())
            {
                double avef = 0.0;
                for (int j = 0; j < atoms[i].size(); ++j)
                {
                    avef += atoms[i].forces_[j].squaredNorm();
                }
                avef /= atoms[i].size();
                atoms[i].fitness *= score(avef, force_dbl, force_ref, force_pow);
            }
            if (sel_by_stress && atoms[i].has_stresses())
            {
                double maxs = 0.0;
                for (int h = 0; h < 3; ++h)
                    for (int k = h; k < 3; ++k)
                    {
                        const double value = atoms[i].stress_(h,k);
                        if (!std::isfinite(value))
                            throw std::invalid_argument("Fitness stress statistic must be finite");
                        maxs = std::max(maxs, std::fabs(value));
                    }
                atoms[i].fitness *= score(maxs, stress_dbl, stress_ref, stress_pow);
            }
            if (!std::isfinite(atoms[i].fitness) || atoms[i].fitness < 0.0 || atoms[i].fitness > 1.0)
                throw std::runtime_error("Fitness must be finite and in [0,1]");
            atoms[i].features_["Fitness"] = to_string(atoms[i].fitness);
            atoms[i].global_attribs_d_["fitness"] = atoms[i].fitness;
            if (fitness_threshold > 0 && atoms[i].fitness > fitness_threshold)
            {
                resatoms.push_back(atoms[i]);
            }
        }
    }

    auto &output = fitness_threshold > 0 ? resatoms : atoms;
    if (!iobj.write(output, outfile, outformat)) {
        cerr << "Error: failed to write " << outfile << '\n';
        return 1;
    }
    Message("Complete!");
    return 0;
} catch (const std::exception &error) {
    cerr << "Error: " << error.what() << '\n';
    return 1;
}
