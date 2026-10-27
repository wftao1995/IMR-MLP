# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Phonopy displacement workflow with an IMR-MLP potential.

Run from the repository root, after installing the optional dependencies:
    python -m pip install '.[phonopy]'
    python examples-python/09_phonopy.py POSCAR model.mtp --elements Al Ni

The structure must already be relaxed for the supplied model. Outputs are
written in the current directory. Phonopy owns cell mapping and postprocessing.
"""
import argparse

from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from imrmlp.calculator import IMRMLPCalculator
from imrmlp.phonopy import calculate_forces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("potential")
    parser.add_argument("--elements", nargs="+", required=True,
                        help="element symbols in model species order")
    parser.add_argument("--potential-type", default="mtp")
    parser.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--mesh", nargs=3, type=int, default=[20, 20, 20])
    args = parser.parse_args()
    if min(args.supercell + args.mesh) <= 0 or args.distance <= 0:
        parser.error("supercell, mesh and distance must be positive")

    atoms = read(args.structure)
    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                           cell=atoms.cell.array,
                           scaled_positions=atoms.get_scaled_positions(),
                           masses=atoms.get_masses())
    phonon = Phonopy(unitcell, supercell_matrix=args.supercell,
                     primitive_matrix="auto", calculator="vasp")
    phonon.generate_displacements(distance=args.distance)
    calculator = IMRMLPCalculator(filename=args.potential,
                                 pottype=args.potential_type,
                                 elements=args.elements)
    calculate_forces(phonon, calculator)
    phonon.produce_force_constants()
    phonon.save("phonopy_params.yaml", settings={"force_constants": True})
    phonon.run_mesh(args.mesh)
    dos = phonon.run_total_dos()
    dos.write()
    thermal = phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
    thermal.write_yaml()
    print("Wrote phonopy_params.yaml, total_dos.dat and thermal_properties.yaml")


if __name__ == "__main__":
    main()
