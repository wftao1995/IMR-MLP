// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

// Standalone test for MTP core
// g++ -std=c++23 -O2 -o test_mtp test_mtp.cpp mtp_kim_core.cpp

#include "mtp_kim_core.h"
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtp_file>" << std::endl;
        return 1;
    }

    mtp_kim::MTPCore core;
    if (!core.load(argv[1])) {
        std::cerr << "Failed to load " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "=== Model loaded successfully ===" << std::endl;
    std::cout << "Species: " << core.speciesCount() << std::endl;
    std::cout << "Cutoff: " << core.max_cutoff << std::endl;
    std::cout << "Alpha moments: " << core.alphaMomentsCount() << std::endl;
    std::cout << "Basic moments: " << core.alphaBasicCount() << std::endl;
    std::cout << "Radial funcs: " << core.radialFuncCount() << std::endl;
    std::cout << "Basis size: " << core.basisSize() << std::endl;

    // Test with a simple 2-atom system
    std::cout << "\n=== Testing 2-atom calculation ===" << std::endl;

    double coords[] = {0.0, 0.0, 0.0,  2.5, 0.0, 0.0};
    int species[] = {0, 0};
    int contrib[] = {1, 1};
    double energy = 0.0;
    double forces[6] = {0};
    double virial[6] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    core.computeEFS(2, species, contrib, coords, energy, forces, virial);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Energy: " << energy << " eV" << std::endl;
    std::cout << "Forces: [" << forces[0] << ", " << forces[1] << ", " << forces[2]
              << "; " << forces[3] << ", " << forces[4] << ", " << forces[5] << "]" << std::endl;
    std::cout << "Virial: [" << virial[0] << ", " << virial[1] << ", " << virial[2]
              << ", " << virial[3] << ", " << virial[4] << ", " << virial[5] << "]" << std::endl;
    std::cout << "Time: " << dur.count() << " us" << std::endl;

    return 0;
}
