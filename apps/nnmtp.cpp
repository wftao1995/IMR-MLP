// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "NNBase/nn1hidden.h"
#include "NNBase/nn2hidden.h"
#include <algorithm>
#include <cmath>
#include <limits>
/**
 * @brief Read potential file, initialize coeffs for NNMTP
 */

using namespace std;

int nnmtp(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    string settingfile = "";
    string mtpfile = "potential.mtp";
    string datafile = "train.cfg";
    string dataformat;
    string index = "index.map";
    string outfile = "potential_specific";
    int n_neuron = 10;
    int n_neuron_2 = 5;
    double safe_factor = 1.0;
    bool initmtp = true;
    bool normalize = true;
    int nnlay = 1;

    bool conv_nn_to_txt = false;
    GPSettings settings;

    cmdopts.addop(settingfile, "settings", "", "Settings file", true)
        .addop(mtpfile, "potfile", "pot", "Potential file")
        .addop(datafile, "data", "d", "training set")
        .addop(dataformat, "input-format", "f", "Input format of training set")
        .addop(n_neuron, "nneuron", "n", "number of hidden neurons")
        .addop(n_neuron_2, "nneuron2", "n2")
        .addop(index, "index", "id")
        .addop(initmtp, "initmtp", "m", "random initialize mtp coeffs")
        .addop(safe_factor, "extent", "e")
        .addop(outfile, "output", "o")
        .addop(nnlay, "nnlay", "l")
        .addop(normalize, "normalize", "norm", "do normalization, default is true")
        .addop(conv_nn_to_txt, "cvnv", "cvnv", "convert .nn to text", false, 0)
        .phrase_options(argc, argv);
    if (!settingfile.empty()) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
    }

    if (nnlay != 1 && nnlay != 2)
        GPERROR("NNMTP --nnlay must be 1 or 2");
    if (conv_nn_to_txt) {
        if (nnlay > 1) {
            IMRMLPNN::NN2lay nn;
            nn.load_coeffs(outfile + ".nn");
            nn.dump_coeffs_txt(outfile + ".nn.txt");
        } else {
            IMRMLPNN::NN1lay nn;
            nn.load_coeffs(outfile + ".nn");
            nn.dump_coeffs_txt(outfile + ".nn.txt");
        }
        return 0;
    }
    if (n_neuron <= 0 || (nnlay == 2 && n_neuron_2 <= 0))
        GPERROR("NNMTP hidden neuron counts must be positive");
    if (normalize && (!std::isfinite(safe_factor) || safe_factor <= 0.0))
        GPERROR("NNMTP --extent must be finite and positive");

    vector<Atoms> atoms;
    iobj.readmap(index);
    if (!iobj.read(atoms, datafile, dataformat))
        GPERROR("Failed to read NNMTP training data: " + datafile);
    if (atoms.empty() || std::any_of(atoms.begin(), atoms.end(), [](const Atoms &at) { return at.size() == 0; }))
        GPERROR("NNMTP requires nonempty training structures");
    Potentials::PotMTP pot(mtpfile);
    pot.omp_init();
    int n_input = pot.alpha_scalar_moments + 1;
    int n_species = pot.species_count;
    for (const auto &at : atoms) {
        if (at.chemmap_.size() != at.size() ||
            std::any_of(at.chemmap_.begin(), at.chemmap_.end(), [n_species](int type) { return type < 0 || type >= n_species; }))
            GPERROR("NNMTP training atom types do not match the MTP species");
    }
    IMRMLPNN::NN1lay nn;
    IMRMLPNN::NN2lay nn2;
    nn.setup(n_input, n_neuron);
    nn.random_init();
    if (nnlay == 2) {
        nn2.setup(n_input, n_neuron, n_neuron_2);
        nn2.random_init();
    }
    if (initmtp) {
        pot.random_init_radial();
        pot.Orthogonalize();
        // monospacing species coeffs from -1 to 1
        for (int i = 0; i < n_species; ++i) {
            pot.species_coeffs[i] = -1.0 + 2. / (n_species + 1.) * (i+1);
        }
    }

    if (normalize) {
        Message("Norm");
        // calculate dist
        vector<double> dmin(pot.alpha_scalar_moments, 1e99), dmax(pot.alpha_scalar_moments, -1e99);
        vector<double> disp(pot.alpha_scalar_moments, 0);
        vector<double> sum(pot.alpha_scalar_moments, 0), sqsum(pot.alpha_scalar_moments, 0);

        size_t site_count = 0;
        for (auto &at : atoms) {
            Neighborhoods nbhs(at, pot.cutoff);
            for (int iat = 0; iat < at.size(); ++iat) {
                auto &nbh = nbhs[iat];
                pot.compute_descriptors(nbh, disp);
                for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
                    if (disp[i+1] < dmin[i]) dmin[i] = disp[i+1];
                    if (disp[i+1] > dmax[i]) dmax[i] = disp[i+1];
                    sum[i] += disp[i+1];
                }
                ++site_count;
            }
        }
        if (site_count == 0) {
            GPERROR("NNMTP normalization requires at least one atom in the training set");
        }
        // get mean
        for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
            sum[i] /= (double) site_count;
        }
        // get sigma
        for (auto &at : atoms) {
            Neighborhoods nbhs(at, pot.cutoff);
            for (int iat = 0; iat < at.size(); ++iat) {
                auto &nbh = nbhs[iat];
                pot.compute_descriptors(nbh, disp);
                for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
                    sqsum[i] += (disp[i+1] - sum[i]) * (disp[i+1] - sum[i]);
                }
            }
        }
        Message("Stat");
        ofstream ofsstat(outfile + ".stat");
        if (!ofsstat) GPERROR("Failed to open normalization statistics: " + outfile + ".stat");
        ofsstat << fmt::format("{:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9}\n", "ID", "MEAN", "STD", "MIN", "MAX", "M/STD", "MIN/STD", "MAX/STD");
        for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
            sqsum[i] /= (double) site_count;
            double stddev = sqrt(sqsum[i]);
            double stat_scale = stddev > std::numeric_limits<double>::epsilon() ? stddev : 1.0;
            ofsstat << fmt::format("{:9} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f}\n",
                                   i+1, sum[i], stddev, dmin[i], dmax[i],
                                   sum[i]/stat_scale, dmin[i]/stat_scale, dmax[i]/stat_scale);
            sqsum[i] = 1.0 / (stat_scale * safe_factor);
            if (!std::isfinite(sum[i]) || !std::isfinite(stddev) ||
                !std::isfinite(sqsum[i]) || sqsum[i] <= 0.0)
                GPERROR("NNMTP normalization produced non-finite or non-positive scales");
        }
        ofsstat.close();
        if (!ofsstat) GPERROR("Failed to write normalization statistics: " + outfile + ".stat");
        Message("Assign");
        for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
            nn.norm_mean(i + 1) = sum[i];
            nn.norm_irstd(i + 1) = sqsum[i];
        }
    }
    if (nnlay == 2) {
        nn2.norm_irstd = nn.norm_irstd;
        nn2.norm_mean = nn.norm_mean;
    }
    // generate species coeffs
    Eigen::MatrixXd matA;
    matA.setZero(n_species, n_species);
    // for (int i = 0; i < n_species; ++i) {
    //     matA(i, i) += 1e-6;
    // }
    Eigen::VectorXd vv, vecA;
    vv.resize(n_species);
    vecA.setZero(n_species);
    size_t unlabeled_count = 0;
    for (auto &at : atoms) {
        if (!at.have_energy) {
            ++unlabeled_count;
            continue;
        }
        double ref_energy = at.energy_;
        if (!std::isfinite(ref_energy)) GPERROR("NNMTP reference energies must be finite");
        double eng = ref_energy / at.size();
        double c = 1.0 / at.size();
        vv.setZero();
        for (int i = 0; i < at.size(); ++i) {
            vv[at.chemmap_[i]] += c;
        }
        for (int i = 0; i < n_species; ++i) {
            for (int j = i; j < n_species; ++j) {
                matA(i, j) += vv[i] * vv[j];
            }
            vecA(i) += vv[i] * eng;
        }
    }
    if (unlabeled_count > 0)
        Warning("Warning: NNMTP skipped " + std::to_string(unlabeled_count) +
                " structures without energy labels when fitting species offsets");
    for (int i = 0; i < n_species; ++i) {
        for (int j = i + 1; j < n_species; ++j) {
            matA(j, i) = matA(i, j);
        }
    }
    if (unlabeled_count == atoms.size()) {
        Warning("Warning: NNMTP has no energy labels; species offsets are initialized to zero");
        vv.setZero();
    } else {
        vv = matA.ldlt().solve(vecA);
        if (!vv.allFinite()) GPERROR("NNMTP species offset fitting produced non-finite coefficients");
    }
    Message("output");
    // output
    pot.Save(outfile + ".mtp");
    if (nnlay > 1)
        nn2.dump_coeffs(outfile + ".nn");
    else
        nn.dump_coeffs(outfile + ".nn");
    ofstream ofs(outfile + ".sc");
    if (!ofs) GPERROR("Failed to open species coefficients: " + outfile + ".sc");
    ofs << fmt::format("{}\n", fmt::join(vv, " "));
    ofs.close();
    if (!ofs) GPERROR("Failed to write species coefficients: " + outfile + ".sc");
    return 0;
}
