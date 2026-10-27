#include "mlip.h"
#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include <cmath>

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 5) { std::cerr << "Usage: model_probe MODEL.mtp [--radial PAIR R]\n"; return 2; }
    try {
        // Exercise more than the historical ten atom types without allocating
        // from an untrusted species_count before the reader validates it.
        std::vector<int> mapping(14, 0);
        mapping[0] = -1;
        double cutoff = 0.;
        auto model = std::unique_ptr<MLIPCore::mlip, decltype(&MLIPCore::destroy_mlip)>(
            MLIPCore::setup(argv[1],mapping.data(),&cutoff,mapping.size()), &MLIPCore::destroy_mlip);
        if (argc == 5) {
            if (std::string(argv[2]) != "--radial") throw std::runtime_error("expected --radial PAIR R");
            const int pair = std::stoi(argv[3]);const double radius=std::stod(argv[4]);
            if (pair<0 || pair>=model->species_count*model->species_count || !std::isfinite(radius))
                throw std::runtime_error("invalid radial probe pair or radius");
            std::unique_ptr<MLIPCore::mlip_buffers,decltype(&MLIPCore::destroy_mlip_buffer)> buffer(
                MLIPCore::create_mlip_buffer(model.get()),MLIPCore::destroy_mlip_buffer);
            if (model->pair_radial_basis_count)
                MLIPCore::RB_CalcDerPair(model.get(),model->pair_radial_bases+pair,buffer.get(),radius);
            else model->rbcalc(model.get(),buffer.get(),radius);
            std::cout << std::setprecision(17) << "{\"values\":[";
            for (int i=0;i<model->rb_size;++i) std::cout << (i?",":"") << buffer->rb_vals[i];
            std::cout << "],\"derivatives\":[";
            for (int i=0;i<model->rb_size;++i) std::cout << (i?",":"") << buffer->rb_ders[i];
            std::cout << "]}\n";
            return 0;
        }
        std::cout << "species=" << model->species_count << " cutoff=" << cutoff
                  << " descriptors=" << model->descriptor_count << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
