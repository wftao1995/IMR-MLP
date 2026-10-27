// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/******************************************
 * Written by wftao (wftao1995@gmail.com)
 * 
 * **************************************/

#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include "utils/gputils.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
std::vector<long double> parse_ratios(const std::string &text)
{
    std::vector<long double> ratios;
    size_t start = 0;
    while (true) {
        const auto end = text.find(',', start);
        auto token = text.substr(start, end == std::string::npos ? end : end - start);
        const auto first = token.find_first_not_of(" \t\r\n\f\v");
        if (first == std::string::npos)
            throw Parameters::Error("--ratios contains an empty value");
        token = token.substr(first, token.find_last_not_of(" \t\r\n\f\v") - first + 1);
        const auto value = Parameters::parse_value<long double>(token);
        if (!std::isfinite(value) || value < 0)
            throw Parameters::Error("--ratios values must be finite and nonnegative");
        ratios.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (ratios.size() < 2)
        throw Parameters::Error("--ratios requires at least two values");
    if (*std::max_element(ratios.begin(), ratios.end()) == 0)
        throw Parameters::Error("--ratios must contain a positive value");
    return ratios;
}

std::vector<size_t> split_sizes(size_t total, std::vector<long double> ratios)
{
    // Scale first so summing large, finite weights cannot overflow.
    const auto maximum = *std::max_element(ratios.begin(), ratios.end());
    for (auto &ratio : ratios) ratio /= maximum;
    const auto sum = std::accumulate(ratios.begin(), ratios.end(), 0.0L);
    std::vector<size_t> sizes(ratios.size()), order(ratios.size());
    std::vector<long double> remainders(ratios.size());
    size_t assigned = 0;
    for (size_t i = 0; i < ratios.size(); ++i) {
        const auto quota = static_cast<long double>(total) * ratios[i] / sum;
        sizes[i] = static_cast<size_t>(std::floor(quota));
        remainders[i] = quota - sizes[i];
        assigned += sizes[i];
    }
    std::iota(order.begin(), order.end(), 0);
    // Stable sorting gives earlier subsets priority when remainders are equal.
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return remainders[a] > remainders[b];
    });
    for (size_t i = 0; i < total - assigned; ++i) ++sizes[order[i]];
    return sizes;
}
}

using namespace std;
int split_set(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    std::string inputfile = "";
    std::string outputfile = "split";
    std::string ratios_arg = "";
    GPIO iobj;
    std::vector<Atoms> atoms;
    int nsplit = 10;
    bool shuffle = false;
    cmdopts.addop(nsplit, "nsplit", "n", "split sets evenly into this many subsets")
           .addop(ratios_arg, "ratios", "", "comma-separated nonnegative relative weights, e.g. 8,1,1; sets the subset count")
           .addop(shuffle, "shuffle", "s", "random shuffle configurations before split")
           .addop(inputfile, "input", "", "input structure sets", true)
           .addop(outputfile, "output-prefix", "", "output file prefix", true)
           .set_usage("Examples:\n  split_set dataset.cfg split_ --nsplit 2\n"
                      "  split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes\n"
                      "Ratio mode takes consecutive segments after any shuffle; equal mode distributes round-robin.\n");
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) return 0;
    if (nsplit <= 0) {
        std::cerr << "Error: nsplit must be positive" << std::endl;
        return 1;
    }
    const bool use_ratios = cmdopts.was_provided("ratios");
    std::vector<long double> ratios;
    if (use_ratios) {
        try {
            ratios = parse_ratios(ratios_arg);
        } catch (const Parameters::Error &error) {
            std::cerr << "Error: " << error.what() << std::endl;
            return 1;
        }
        if (cmdopts.was_provided("nsplit") && static_cast<size_t>(nsplit) != ratios.size()) {
            std::cerr << "Error: nsplit must match the number of --ratios values" << std::endl;
            return 1;
        }
    }
    iobj.setmap();
    if (!GPIO::file_exist(inputfile))
    {
        std::cerr << "Error: inputfile " << inputfile << " not found" << std::endl;
        std::exit(128);
    }
    if (!iobj.read(atoms, inputfile)) return 1;
    if (shuffle) {
        std::random_device rd;
        std::mt19937 mtd(rd());
        std::ranges::shuffle(atoms, mtd);
    }
    std::vector<std::vector<Atoms>> splits(use_ratios ? ratios.size() : static_cast<size_t>(nsplit));
    if (use_ratios) {
        const auto sizes = split_sizes(atoms.size(), ratios);
        size_t offset = 0;
        for (size_t i = 0; i < splits.size(); ++i) {
            splits[i].reserve(sizes[i]);
            for (size_t j = 0; j < sizes[i]; ++j)
                splits[i].push_back(std::move(atoms[offset++]));
        }
    } else {
        for (size_t i = 0; i < atoms.size(); ++i)
            splits[i % splits.size()].push_back(std::move(atoms[i]));
    }
    for (size_t i = 0; i < splits.size(); ++i)
    {
        const auto filename = outputfile + std::to_string(i) + ".xyz";
        if (use_ratios && splits[i].empty()) {
            auto output = open_ofile(filename);
            if (!output || !close_ofile(output)) {
                std::cerr << "Error: cannot write empty subset " << filename << std::endl;
                return 1;
            }
        } else if (!iobj.write(splits[i], filename, "extxyz")) return 1;
    }

    return 0;
}
