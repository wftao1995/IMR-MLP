// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "io/gpio_tokenizer.h"
#include "apps/mlpphdisp_utils.h"

using namespace std;
using namespace Eigen;

namespace phonopyfc_detail {

void read_required_yaml_line(std::istream &input,
                             std::string &line,
                             std::string_view expected,
                             std::string_view filename)
{
    if (!std::getline(input, line)) {
        throw std::runtime_error(fmt::format(
            "phonopyfc: expected '{}' before end of YAML file '{}'",
            expected, filename));
    }
}

void find_required_yaml_field(std::istream &input,
                              std::string &line,
                              std::string_view field,
                              std::string_view filename)
{
    while (std::getline(input, line)) {
        if (line.starts_with(field))
            return;
    }
    throw std::runtime_error(fmt::format(
        "phonopyfc: required field '{}' not found in YAML file '{}'",
        field, filename));
}

} // namespace phonopyfc_detail

namespace {

void ensure_yaml_open(const std::ifstream &input, const std::string &filename)
{
    if (!input.is_open()) {
        throw std::runtime_error(
            fmt::format("phonopyfc: cannot open YAML file '{}'", filename));
    }
}

map<string, int> read_species_map(const string &filename)
{
    GPIO io;
    if (!io.readmap(filename))
        throw std::runtime_error("phonopyfc: Cannot open index file or read species map: " + filename);
    return io.getrevmap();
}

int species_index(const map<string, int> &index, const string &symbol,
                  const Potentials::AnyPotential &pot)
{
    const auto it = index.find(symbol);
    if (it == index.end())
        throw std::runtime_error("phonopyfc: unknown element in index map: " + symbol);
    if (it->second < 0 || it->second >= pot.species_count)
        throw std::runtime_error("phonopyfc: species index outside potential range: " + symbol);
    return it->second;
}

RowVector3d &displaced_position(Atoms &atom, int index, const string &filename)
{
    if (index < 1 || index > atom.size())
        throw std::runtime_error(fmt::format(
            "phonopyfc: displacement atom {} is outside 1..{} in '{}'",
            index, atom.size(), filename));
    return atom.pos_[index - 1];
}

void close_force_output(ofstream &output, const string &filename)
{
    output.close();
    if (!output)
        throw std::runtime_error("phonopyfc: failed to write " + filename);
}

} // namespace

void mlpphdisp_splitstr(string &instr, vector<string> &vecstr)
{
    vecstr.clear();
    std::stringstream ss1(instr);
    std::string word;
    while (ss1 >> word)
        vecstr.push_back(word);
}

void ReadPhonoYaml(vector<Atoms> &cfgs, vector<int> &dispid, vector<RowVector3d> &dispvec, Potentials::AnyPotential *pot, string indexfilename, string yamlname)
{
    string line;
    const auto index = read_species_map(indexfilename);
    vector<string> vecstr0;
    string word0;

    //read supercell
    ifstream ifs(yamlname);
    ensure_yaml_open(ifs, yamlname);
    phonopyfc_detail::find_required_yaml_field(
        ifs, line, "supercell:", yamlname);
    getline(ifs, line); //lattice
    Matrix3d cell;
    vector<RowVector3d> pos;
    vector<int> types;
    for (int i = 0; i < 3; i++)
    {
        getline(ifs, line);
        mlpphdisp_splitstr(line, vecstr0);
        //2,3,4;
        cell(i, 0) = stod(vecstr0.at(2).substr(0, vecstr0.at(2).length() - 1));
        cell(i, 1) = stod(vecstr0.at(3).substr(0, vecstr0.at(3).length() - 1));
        cell(i, 2) = stod(vecstr0.at(4));
    }
    getline(ifs, line); //points:
    do
    {
        getline(ifs, line);      // symbol
        mlpphdisp_splitstr(line, vecstr0); // in 2
        if (vecstr0.size() == 0)
            break;
        word0 = vecstr0.at(2);
        types.push_back(species_index(index, word0, *pot));
        getline(ifs, line); // coordinates, 2 3 4
        mlpphdisp_splitstr(line, vecstr0);
        RowVector3d tmpvec, tmppos;
        tmpvec[0] = stod(vecstr0.at(2).substr(0, vecstr0.at(2).length() - 1));
        tmpvec[1] = stod(vecstr0.at(3).substr(0, vecstr0.at(3).length() - 1));
        tmpvec[2] = stod(vecstr0.at(4));
        tmppos = tmpvec * cell;
        pos.push_back(tmppos);
        getline(ifs, line); //mass
        getline(ifs, line); //reduced_to
    }while(line.substr(0,14) != "displacements:");
    getline(ifs, line); // displacements
    int wdisp;
    while (!ifs.eof())
    {
        getline(ifs, line);      // atom
        mlpphdisp_splitstr(line, vecstr0); // 2
        if(vecstr0.size() <2)
            break;
        wdisp = stoi(vecstr0.at(2));
        dispid.push_back(wdisp);
        getline(ifs, line);
        getline(ifs, line);
        mlpphdisp_splitstr(line, vecstr0);
        RowVector3d tmpvec;
        tmpvec[0] = stod(vecstr0.at(1).substr(0, vecstr0.at(1).length() - 1));
        tmpvec[1] = stod(vecstr0.at(2).substr(0, vecstr0.at(2).length() - 1));
        tmpvec[2] = stod(vecstr0.at(3));
        dispvec.push_back(tmpvec);
        Atoms cfg;
        cfg.resize(types.size());
        cfg.cell_ = cell;
        for (int i = 0; i < cfg.size(); i++)
        {
            cfg.chemmap_[i] = types[i];
            cfg.pos_[i] = pos[i];
        }
        displaced_position(cfg, wdisp, yamlname) += tmpvec;
        cfg.wrap();
        pot->getEFS(cfg);
        cfgs.push_back(cfg);
    }
}

void SaveForcesSet(vector<Atoms> &cfgs, vector<int> &dispid, vector<RowVector3d> &dispvec)
{
    if (cfgs.empty())
        throw std::runtime_error("phonopyfc: no displacement configurations");
    string output = "FORCE_SETS";
    ofstream ofs(output);
    if (!ofs) throw std::runtime_error("phonopyfc: cannot open " + output);
    fmt::print(ofs, "{}\n{}\n\n", cfgs[0].size(), dispid.size());
    for (int i = 0; i < dispid.size(); i++)
    {
        fmt::print(ofs, "{}\n{: 20.16f}\n", dispid[i], fmt::join(dispvec[i], " "));
        for (int j = 0; j < cfgs[i].size(); j++)
            fmt::print(ofs, "  {:23.16g}\n", fmt::join(cfgs[i].forces_[j], " "));
        fmt::print(ofs, "\n");
    }
    close_force_output(ofs, output);
}

int phonopyfc(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    string indexfilename = "index.map";
    string settingfile = "settings.ini";
    GPSettings settings;
    string yamlname = "phonopy_disp.yaml";
    vector<Atoms> cfgs;
    vector<int> dispid;
    vector<RowVector3d> dispvec;
    string potfn = "potential.mtp";
    cmdopts.addop(indexfilename, "index-file", "id", "index file")
           .addop(settingfile, "settings-file", "i", "settings file")
           .addop(potfn, "potfile", "pot", "Potential file", true)
           .addop(yamlname, "disp", "d", "displament yaml file generated by phonopy", true);
    cmdopts.phrase_options(argc, argv);           
    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        fmt::print("Settings loaded from file: {}\n", settingfile);
    }
    apply_settings_to_cmdopts(cmdopts, settings);

    Potentials::AnyPotential *pot;
    Potentials::potential_wrapper wrapper(potfn, "mtp", settings.ExtractSubSettings("pot"));
    pot = wrapper.pot;
    ReadPhonoYaml(cfgs, dispid, dispvec, pot, indexfilename, yamlname);
    SaveForcesSet(cfgs, dispid, dispvec);
    
    return 0;
}
