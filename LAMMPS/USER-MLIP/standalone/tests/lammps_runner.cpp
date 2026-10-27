#include "register.h"
#include "lammps.h"
#include "input.h"
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto *lmp = new LAMMPS_NS::LAMMPS(argc, argv, MPI_COMM_WORLD);
    register_mtp_standalone(lmp);
    lmp->input->file();
    delete lmp;
    MPI_Finalize();
}
