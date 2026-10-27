#include "mlip_model_mpi.h"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0, failures = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2 || size < 2) {
        if (rank == 0) std::cerr << "Usage: mpiexec -n 2|4 test_model_mpi MISSING_MODEL_PATH\n";
        MPI_Finalize();
        return 2;
    }
    bool ran = false;
    try { MLIPCore::collective_try(MPI_COMM_WORLD, [&] { ran = true; }); }
    catch (...) { ++failures; }
    if (!ran) ++failures;

    bool caught = false;
    try {
        MLIPCore::collective_try(MPI_COMM_WORLD, [&] {
            if (rank == 1) throw std::runtime_error("rank-one-initialization-failure");
        });
    } catch (const std::runtime_error &error) {
        caught = std::string(error.what()) == "rank-one-initialization-failure";
    } catch (...) { ++failures; }
    if (!caught) ++failures;

    caught = false;
    try { (void) MLIPCore::read_model_text_mpi(MPI_COMM_WORLD, argv[1]); }
    catch (const std::runtime_error &error) { caught = error.what()[0] != '\0'; }
    catch (...) { ++failures; }
    if (!caught) ++failures;

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "MPI error propagation: ranks=" << size << " failures=" << total << '\n';
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
