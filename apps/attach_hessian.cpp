// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "apps.h"
#include "io/gpio.h"
#include "tools/hessian_io.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>

namespace {

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string guess_source(const std::string &filename)
{
    const std::string lower = lower_copy(filename);
    if (lower.find("vasprun") != std::string::npos || lower.find(".xml") != std::string::npos)
        return "dfpt";
    return "fd";
}

}

int attach_hessian(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO io;
    std::vector<Atoms> atoms;

    std::string infile = "POSCAR";
    std::string informat = "";
    std::string outfile = "hessian.xyz";
    std::string outformat = "extxyz";
    std::string hessian_file = "";
    std::string source = "auto";
    std::string vasprun_key = "auto";
    int frame = -1;
    double scale = 1.0;
    bool symmetrize = false;
    int hvp_probes = 0, hvp_seed = 42;
    bool hvp_only = false;

    cmdopts.addop(infile, "input", "in", "Input structure file", true)
           .addop(outfile, "output", "out", "Output extxyz file", true)
           .addop(informat, "input-format", "if", "Input structure format")
           .addop(outformat, "output-format", "of", "Output format, default extxyz")
           .addop(hessian_file, "hessian", "", "DFPT vasprun.xml or FD FORCE_CONSTANTS file")
           .addop(source, "source", "s", "Hessian source: auto, dfpt, or fd")
           .addop(vasprun_key, "vasprun-key", "", "vasprun.xml varray name, default auto")
           .addop(frame, "frame", "f", "Structure frame index; -1 uses the last frame")
           .addop(scale, "scale", "", "Multiply Hessian by this factor before writing")
           .addop(symmetrize, "symmetrize", "", "Replace H by 0.5*(H+H^T)", false, 0)
           .addop(hvp_probes, "hvp-probes", "", "Number of random Hessian-vector labels (default 0)")
           .addop(hvp_seed, "hvp-seed", "", "Random seed for HVP directions")
           .addop(hvp_only, "hvp-only", "", "Store only HVP labels, omitting the dense Hessian", false, 0)
           .add_request("input")
           .add_request("output")
           .set_helponexit(false)
           .phrase_options(argc, argv);

    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        return 0;
    }

    try {
        if (hvp_probes < 0 || (hvp_only && hvp_probes == 0))
            throw std::invalid_argument("--hvp-only requires --hvp-probes > 0");
        if (!io.read(atoms, infile, informat))
            throw std::runtime_error("Failed to read input structures");
        if (atoms.empty())
            throw std::runtime_error("No structures read from input: " + infile);

        int iframe = frame;
        if (iframe < 0)
            iframe = static_cast<int>(atoms.size()) - 1;
        if (iframe < 0 || iframe >= static_cast<int>(atoms.size()))
            throw std::runtime_error("Frame index out of range: " + std::to_string(frame));

        Atoms atom = atoms[iframe];
        if (hessian_file.empty())
            hessian_file = infile;
        if (source == "auto")
            source = guess_source(hessian_file);
        source = lower_copy(source);

        Eigen::MatrixXd hessian;
        if (source == "fd") {
            hessian = HessianIO::read_force_constants(hessian_file, atom.size());
        } else if (source == "dfpt") {
            hessian = HessianIO::read_vasprun_hessian(hessian_file, atom.size(), vasprun_key);
        } else {
            throw std::runtime_error("Unsupported --source '" + source + "', expected auto, dfpt, or fd");
        }

        if (scale != 1.0)
            hessian *= scale;
        if (symmetrize)
            hessian = 0.5 * (hessian + hessian.transpose()).eval();

        atom.hessian_ = std::move(hessian);
        atom.have_hessian = true;
        atom.clear_hvp();
        if (hvp_probes > 0) {
            if (atom.size() < 2) throw std::invalid_argument("HVP probes require at least two atoms");
            std::mt19937 random(hvp_seed);
            std::normal_distribution<double> normal;
            atom.hvp_directions_.resize(3 * atom.size(), hvp_probes);
            for (int k = 0; k < hvp_probes; ++k) {
                auto direction = atom.hvp_directions_.col(k);
                for (int i = 0; i < direction.size(); ++i) direction[i] = normal(random);
                Eigen::Vector3d mean = Eigen::Vector3d::Zero();
                for (int i = 0; i < atom.size(); ++i) mean += direction.segment<3>(3 * i);
                mean /= atom.size();
                for (int i = 0; i < atom.size(); ++i) direction.segment<3>(3 * i) -= mean;
                direction.normalize();
            }
            atom.hvp_targets_.noalias() = atom.hessian_ * atom.hvp_directions_;
            atom.validate_hvp();
        }
        if (hvp_only) { atom.hessian_.resize(0, 0); atom.have_hessian = false; }

        std::vector<Atoms> output_atoms{atom};
        if (!io.write(output_atoms, outfile, outformat))
            throw std::runtime_error("Failed to write Hessian/HVP dataset");
    } catch (const std::exception &err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
        return 128;
    }

    return 0;
}
