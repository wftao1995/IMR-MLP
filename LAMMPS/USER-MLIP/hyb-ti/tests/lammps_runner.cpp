// Regression runner: register the pair style without modifying upstream LAMMPS.
#include "lammps.h"
#include "force.h"
#include "input.h"
#include "pair_harmonic_reference.h"
#include <mpi.h>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  {
    LAMMPS_NS::LAMMPS lmp(argc, argv, MPI_COMM_WORLD);
    (*lmp.force->pair_map)["harmonic/reference"] = [](LAMMPS_NS::LAMMPS *p)
        -> LAMMPS_NS::Pair * { return new LAMMPS_NS::PairHarmonicReference(p); };
    lmp.input->file();
  }
  MPI_Finalize();
}
