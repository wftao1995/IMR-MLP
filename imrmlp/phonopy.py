# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Evaluate Phonopy displacement forces with an ASE calculator (optional deps)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from phonopy import Phonopy


def calculate_forces(phonon: Phonopy, calculator: Calculator) -> np.ndarray:
    """Set and return ``phonon.forces`` in eV/Angstrom.

    Generate displacements with Phonopy first. The Phonopy calculator must be
    unset or ``"vasp"`` (Angstrom/eV units). Supercell and displacement ordering
    are preserved; forces are neither drift-corrected nor symmetrized. Existing
    forces are replaced only after every calculation succeeds. Call Phonopy's
    ``produce_force_constants()`` and analysis methods afterwards.
    """
    from ase import Atoms

    if phonon.calculator not in (None, "vasp"):
        raise ValueError("Use Phonopy calculator=None or 'vasp' for Angstrom/eV units")
    supercells = phonon.supercells_with_displacements
    if not supercells or any(cell is None for cell in supercells):
        raise ValueError("Generate a complete set of Phonopy displacements first")

    forces = []
    for cell in supercells:
        atoms = Atoms(symbols=cell.symbols, cell=cell.cell,
                      scaled_positions=cell.scaled_positions,
                      masses=cell.masses, pbc=True, calculator=calculator)
        values = np.asarray(atoms.get_forces(), dtype=float)
        if values.shape != (len(cell), 3) or not np.isfinite(values).all():
            raise ValueError("Calculator forces must be finite with shape (N, 3)")
        forces.append(values.copy())
    result = np.array(forces)
    phonon.forces = result
    return result
