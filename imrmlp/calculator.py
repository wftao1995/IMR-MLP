# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
ASE Calculator integration for imrmlp potentials.

Provides IMRMLPCalculator — a drop-in ASE Calculator that wraps any
imrmlp potential (MTP, EAM, LJ, etc.) for use with ASE's ecosystem
(optimizers, MD, NEB, vibration analysis, etc.).

Usage:
    from imrmlp.calculator import IMRMLPCalculator
    from ase.build import bulk
    from ase.optimize import BFGS

    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    atoms.calc = IMRMLPCalculator(filename="Al.mtp")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    opt = BFGS(atoms)
    opt.run(fmax=0.01)
"""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np

from ase.calculators.calculator import (
    Calculator, PropertyNotImplementedError, all_changes,
)

from .potential import Potential
from .atoms import Atoms
from .utils import build_type_mapping


class IMRMLPCalculator(Calculator):
    """ASE Calculator wrapping a imrmlp potential.

    Supports all potential types available in imrmlp (MTP, EAM, LJ, SW, etc.).
    Per-atom ``energies`` are available when the potential returns a finite
    value for every atom whose sum agrees with its total energy.
    Atomic ``charges`` are available when the potential produces one finite
    computed charge per atom, independently of ASE initial charges.

    Parameters
    ----------
    filename : str, optional
        Path to the potential file.
    potential : Potential, optional
        An existing imrmlp.Potential object.
    pottype : str
        Potential type (default: "mtp"). Only used when filename is given.
    elements : list of str, optional
        Element symbols in chemmap-index order, e.g. ['Al', 'Mg'].
        Required if the potential's species mapping differs from atomic numbers.

    Examples
    --------
    >>> from imrmlp.calculator import IMRMLPCalculator
    >>> from ase.build import bulk
    >>>
    >>> atoms = bulk("Al", "fcc", a=4.05)
    >>> atoms.calc = IMRMLPCalculator(filename="Al.mtp")
    >>> e = atoms.get_potential_energy()
    """

    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'charges']

    def __init__(self,
                 filename: Optional[str] = None,
                 potential: Optional[Potential] = None,
                 pottype: str = "auto",
                 elements: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)

        if potential is not None:
            self._pot = potential
        elif filename is not None:
            self._pot = Potential(str(filename), pottype)
        else:
            raise ValueError(
                "Either 'filename' or 'potential' must be provided")

        self._elements = None if elements is None else list(elements)
        self._type_map: Optional[Dict[int, int]] = None
        if self._elements is not None:
            self._type_map = build_type_mapping(self._elements)

        self._imrmlp_atoms: Optional[Atoms] = None

    @staticmethod
    def _effective_total_charge(atoms):
        value = atoms.info.get('total_charge')
        return None if value is None else float(value)

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None and (
                self._effective_total_charge(self.atoms)
                != self._effective_total_charge(atoms)):
            changes.append('total_charge')
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Compute energy alone or full EFS for the requested properties.

        Called automatically by ASE.
        """
        if properties is None:
            properties = ['energy', 'forces', 'stress']

        super().calculate(atoms, properties, system_changes)
        self.results = {}

        # Fresh native outputs prevent a missing property from retaining values
        # from an earlier calculation, including explicit same-geometry calls.
        self._imrmlp_atoms = self._ase_to_imrmlp(self.atoms)
        imrmlp = self._imrmlp_atoms
        if imrmlp is None:
            raise RuntimeError("Internal error: imrmlp_atoms not initialized")

        # Native energy evaluation also supplies per-atom energies.
        if set(properties) <= {'energy', 'energies'}:
            self._pot.calculate(imrmlp, properties=('energy',))
        else:
            self._pot.calculate(imrmlp)

        # Extract results
        if imrmlp.energy is not None:
            self.results['energy'] = float(imrmlp.energy)

        site_energies = imrmlp._site_energies_for_ase()
        if site_energies is not None:
            self.results['energies'] = site_energies
        elif 'energies' in properties:
            raise PropertyNotImplementedError(
                "The potential did not return complete finite per-atom energies "
                "consistent with the total energy")

        charges = imrmlp.charges
        if (charges is not None and charges.shape == (len(imrmlp),)
                and np.all(np.isfinite(charges))):
            self.results['charges'] = np.array(charges, dtype=np.float64, copy=True)

        if 'forces' in properties and imrmlp.forces is not None:
            self.results['forces'] = np.asarray(imrmlp.forces, dtype=np.float64)

        if 'stress' in properties and imrmlp.stress is not None:
            # Native virial = -dE/dstrain; ASE uses +dE/dstrain / volume.
            s = imrmlp.stress
            volume = self.atoms.get_volume()
            self.results['stress'] = -np.array([
                s[0, 0], s[1, 1], s[2, 2],
                s[1, 2], s[0, 2], s[0, 1]
            ], dtype=np.float64) / volume

    def _ase_to_imrmlp(self, atoms) -> Atoms:
        """Convert ASE Atoms to imrmlp Atoms."""
        numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        cell = atoms.get_cell().array

        imrmlp_atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=atoms.get_pbc().tolist(),
            elements=self._elements,
        )
        imrmlp_atoms.total_charge = self._effective_total_charge(atoms)
        return imrmlp_atoms

    def __repr__(self) -> str:
        return f"IMRMLPCalculator(potential={self._pot})"
