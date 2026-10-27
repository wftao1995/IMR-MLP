#include "unitest/maxvol_mtp_fixture.h"
#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int status = 0;
    try {
        if (argc != 2) throw std::runtime_error("Usage: generate_family_models DIRECTORY");
        const auto dir = std::filesystem::absolute(argv[1]);
        const auto cases = make_maxvol_mtp_models(dir);
        std::ofstream manifest(dir / "models.tsv");
        for (const auto &test : cases) {
            const auto settings_path = dir / (test.name + ".ini");
            std::ofstream settings(settings_path);
            settings << "builder:dtype pot_efs\nbuilder:pottype " << test.pottype
                     << "\nbuilder:potfile " << test.path
                     << "\nbuilder:w-energy 1\nbuilder:w-force 0.01\nbuilder:w-stress 0.001\n"
                     << "selector:select-mode full\nselector:init-mode data\n";
            for (const auto &[key, value] : test.settings.settings)
                settings << "pot:" << key << ' ' << value << '\n';
            manifest << test.name << '\t' << test.pottype << '\t' << test.path
                     << '\t' << settings_path.string() << '\n';
        }
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
