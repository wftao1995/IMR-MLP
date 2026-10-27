# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Pythonic Atoms wrapper around the C++ _Atoms class.

Provides:
  - Clean property names (numbers, positions, cell, energy, forces, stress)
  - Automatic atomic_number <-> chemmap index translation
  - ASE interop (to_ase / from_ase)
  - Rich repr for Jupyter notebooks
"""

from __future__ import annotations
from typing import Optional, Dict, List, Sequence, Union
import numpy as np

from ._imrmlp import _Atoms
from .utils import Z_TO_SYMBOL, SYMBOL_TO_Z, chemmap_from_numbers, numbers_from_chemmap


class Atoms:
    """Atomic structure container.

    Wraps the C++ Atoms object with a Pythonic interface.
    Atomic positions are in Angstrom, Cartesian by default.

    Parameters
    ----------
    numbers : numpy.ndarray, optional
        Atomic numbers per atom (shape (N,), dtype int).
    positions : numpy.ndarray, optional
        Atomic positions in Cartesian coordinates (shape (N, 3), dtype float).
    cell : numpy.ndarray, optional
        Unit cell matrix (shape (3, 3), row-major, dtype float).
    pbc : bool or sequence of three bool, optional
        Periodic boundary flags along the three lattice directions. Defaults
        to fully periodic for compatibility with existing IMRMLP structures.
    elements : list of str, optional
        Element symbols in chemmap-index order, e.g. ['Al', 'Mg'].
        Used to translate numbers <-> chemmap.
    _impl : _Atoms, optional
        Existing C++ Atoms object to wrap (internal use).

    Examples
    --------
    >>> atoms = Atoms(
    ...     numbers=np.array([13, 13, 13, 13]),
    ...     positions=np.array([[0,0,0], [1.4,1.4,0], [2.8,0,1.4], [1.4,2.8,1.4]]),
    ...     cell=np.eye(3) * 4.05,
    ... )
    >>> print(atoms.natoms)
    4
    """

    def __init__(self,
                 numbers: Optional[np.ndarray] = None,
                 positions: Optional[np.ndarray] = None,
                 cell: Optional[np.ndarray] = None,
                 pbc: Optional[Union[bool, Sequence[bool]]] = None,
                 elements: Optional[List[str]] = None,
                 _impl: Optional[_Atoms] = None):
        if _impl is not None:
            self._impl = _impl
        else:
            self._impl = _Atoms()
        self._elements: Optional[List[str]] = None if elements is None else list(elements)
        self._reverse_map: Optional[Dict[int, str]] = None

        if self._elements is not None:
            self._reverse_map = {i: el for i, el in enumerate(self._elements)}

        if _impl is None:
            self.pbc = True if pbc is None else pbc
        elif pbc is not None:
            self.pbc = pbc

        if numbers is not None:
            self.numbers = np.asarray(numbers, dtype=np.int32)
        if positions is not None:
            self.positions = np.asarray(positions, dtype=np.float64)
        if cell is not None:
            self.cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)

    # ── numbers <-> chemmap ──────────────────────────────────────────

    @property
    def numbers(self) -> np.ndarray:
        """Atomic numbers (shape (N,), dtype int32)."""
        chem = self._impl.chemmap
        if self._reverse_map is not None:
            return numbers_from_chemmap(chem, self._reverse_map)
        # No mapping info: return chemmap as-is (may be species indices)
        return np.array(chem, dtype=np.int32)

    @numbers.setter
    def numbers(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.int32)
        if self._elements is not None:
            self._impl.chemmap = chemmap_from_numbers(values, self._elements)
        else:
            # Without elements info, use atomic numbers directly as chemmap.
            # This preserves the actual atomic numbers in the C++ layer.
            self._impl.chemmap = [int(v) for v in values]

    @property
    def chemmap(self) -> list:
        """Raw species index per atom (from C++ layer)."""
        return self._impl.chemmap

    @chemmap.setter
    def chemmap(self, values):
        self._impl.chemmap = list(values)

    @property
    def hessian(self) -> np.ndarray:
        """Cartesian Hessian labels (3*N, 3*N), eV/Angstrom^2; empty if absent."""
        return self._impl.hessian

    def set_hessian(self, target: np.ndarray):
        """Set all finite Cartesian Hessian entries, without symmetrizing them."""
        self._impl.set_hessian(np.asarray(target, dtype=np.float64))

    def clear_hessian(self):
        """Remove all full Hessian labels."""
        self._impl.clear_hessian()

    @property
    def hvp_directions(self) -> np.ndarray:
        """Dimensionless Cartesian probes, shape (3*N, m), atom-major xyz rows."""
        return self._impl.hvp_directions

    @property
    def hvp_targets(self) -> np.ndarray:
        """Reference Hessian-vector products in eV/Angstrom^2, shape (3*N, m)."""
        return self._impl.hvp_targets

    def set_hvp(self, directions: np.ndarray, targets: np.ndarray):
        """Set paired HVP labels; reject invalid shapes, nonfinite or zero probes."""
        self._impl.set_hvp(np.asarray(directions, dtype=np.float64),
                           np.asarray(targets, dtype=np.float64))

    def clear_hvp(self):
        """Remove all HVP labels."""
        self._impl.clear_hvp()

    # ── Geometry ─────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions (shape (N, 3), Cartesian by default)."""
        return self._impl.positions

    @positions.setter
    def positions(self, value: np.ndarray):
        self._impl.positions = np.asarray(value, dtype=np.float64)

    @property
    def selective_dynamics_mask(self) -> Optional[np.ndarray]:
        """Owned Boolean (N, 3) atomic mask, or None when absent.

        True/1 permits movement and False/0 fixes the corresponding active
        optimizer coordinate. Use with ``Relaxation(selective_dynamics=True)``.
        The existing fixed-cell default uses Cartesian components, including
        slanted cells. Other coordinate modes retain their native convention.
        This does not set a cell mask or import ASE constraints.
        """
        return self._impl.selective_dynamics_mask

    @selective_dynamics_mask.setter
    def selective_dynamics_mask(self, value: Optional[np.ndarray]):
        if value is None:
            self._impl.selective_dynamics_mask = None
            return
        values = np.asarray(value)
        if values.shape != (len(self), 3):
            raise ValueError("Selective dynamics mask must have shape (N, 3)")
        if np.iscomplexobj(values):
            raise ValueError("Selective dynamics mask must contain only binary 0 or 1 values")
        self._impl.selective_dynamics_mask = np.asarray(values, dtype=np.float64)

    @property
    def cell(self) -> np.ndarray:
        """Unit cell matrix (3x3, row-major)."""
        return self._impl.cell

    @cell.setter
    def cell(self, value: np.ndarray):
        self._impl.cell = np.asarray(value, dtype=np.float64).reshape(3, 3)

    @property
    def pbc(self) -> np.ndarray:
        """Periodic boundary flags along the three lattice directions."""
        return np.asarray(self._impl.pbc, dtype=bool)

    @pbc.setter
    def pbc(self, value):
        flags = np.asarray([value] * 3 if isinstance(value, (bool, np.bool_)) else value,
                           dtype=bool)
        if flags.shape != (3,):
            raise ValueError("pbc must be a bool or a length-3 sequence")
        self._impl.pbc = [bool(v) for v in flags]

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return self._impl.size()

    @property
    def volume(self) -> float:
        """Cell volume (Angstrom^3)."""
        return self._impl.get_volume()

    # ── Physical properties ──────────────────────────────────────────

    @property
    def energy(self) -> Optional[float]:
        """Total energy (eV), or None if not computed."""
        return self._impl.energy if self._impl.has_energy else None

    @energy.setter
    def energy(self, value: float):
        self._impl.energy = value

    @property
    def free_energy(self) -> Optional[float]:
        """Stored free energy (eV), independent of total energy, or None."""
        return self._impl.free_energy

    @free_energy.setter
    def free_energy(self, value: Optional[float]):
        """Set a finite free energy, or clear its presence with None."""
        self._impl.free_energy = None if value is None else float(value)

    @property
    def forces(self) -> Optional[np.ndarray]:
        """Atomic forces (N, 3) in eV/A, or None if not computed."""
        if self._impl.has_forces:
            return self._impl.forces
        return None

    @forces.setter
    def forces(self, value: np.ndarray):
        self._impl.forces = np.asarray(value, dtype=np.float64)

    @property
    def stress(self) -> Optional[np.ndarray]:
        """Stress tensor (3, 3) in eV, or None if not computed."""
        if self._impl.has_stress:
            return self._impl.stress
        return None

    @stress.setter
    def stress(self, value: np.ndarray):
        self._impl.stress = np.asarray(value, dtype=np.float64).reshape(3, 3)

    @property
    def site_energies(self) -> Optional[np.ndarray]:
        """Per-atom site energies (eV)."""
        if self._impl.has_site_energies:
            return np.array(self._impl.site_energies)
        return None

    @property
    def charges(self) -> Optional[np.ndarray]:
        """Computed per-atom charges (e), as an independent array, or None."""
        values = self._impl.charges
        return np.array(values, dtype=np.float64) if values else None

    @property
    def magmoms(self) -> Optional[np.ndarray]:
        """Computed magnetic moments (mu_B), as an independent (N,) or (N, 3) array.

        Missing moments return None. Simultaneous native scalar and vector
        representations cannot be exported through this single property.
        """
        return self._impl.magmoms

    @magmoms.setter
    def magmoms(self, values: Optional[np.ndarray]):
        """Set finite scalar/vector moments, replacing the other representation.

        None clears both representations. No magnetic axis or total moment is
        inferred from per-atom values.
        """
        self._impl.magmoms = None if values is None else np.asarray(values, dtype=np.float64)

    @property
    def total_charge(self) -> Optional[float]:
        """Explicit total charge (e), independent of atomic charges, or None."""
        return self._impl.total_charge

    @total_charge.setter
    def total_charge(self, value: Optional[float]):
        """Set a finite total charge, or remove the explicit value with None."""
        self._impl.total_charge = None if value is None else float(value)

    @property
    def masses(self) -> Optional[np.ndarray]:
        """Explicit per-atom masses in atomic mass units, or None."""
        return self._impl.masses

    @masses.setter
    def masses(self, value: Optional[np.ndarray]):
        self._impl.masses = value

    @property
    def tags(self) -> Optional[np.ndarray]:
        """Explicit integer atom tags (N,), or None; independent of results."""
        return self._impl.tags

    @tags.setter
    def tags(self, value: Optional[np.ndarray]):
        self._impl.tags = value

    @property
    def initial_charges(self) -> Optional[np.ndarray]:
        """Explicit initial charges (N,) in e, separate from computed charges."""
        return self._impl.initial_charges

    @initial_charges.setter
    def initial_charges(self, value: Optional[np.ndarray]):
        self._impl.initial_charges = value

    @property
    def initial_magmoms(self) -> Optional[np.ndarray]:
        """Explicit initial moments (N,) or (N, 3) in mu_B, or None."""
        return self._impl.initial_magmoms

    @initial_magmoms.setter
    def initial_magmoms(self, value: Optional[np.ndarray]):
        self._impl.initial_magmoms = value

    @property
    def velocities(self) -> Optional[np.ndarray]:
        """Cartesian atomic velocities (N, 3) in Angstrom/ps, or None."""
        if self._impl.has_velocities:
            return self._impl.velocities
        return None

    @velocities.setter
    def velocities(self, value: Optional[np.ndarray]):
        """Set Cartesian velocities in Angstrom/ps, or clear them with None."""
        self._impl.velocities = (np.empty((0, 3)) if value is None
                                 else np.asarray(value, dtype=np.float64))

    # ── Methods ──────────────────────────────────────────────────────

    def wrap(self):
        """Wrap atoms back into the periodic cell."""
        self._impl.wrap()

    def copy(self) -> "Atoms":
        """Return a deep copy."""
        return Atoms(
            _impl=self._impl.copy(),
            elements=self._elements,
            pbc=self.pbc,
        )

    def __copy__(self) -> "Atoms":
        return self.copy()

    def to_fractional(self):
        """Convert positions to fractional coordinates; no-op if already fractional."""
        if self._impl.cartesian:
            self._impl.to_fractional()

    def to_cartesian(self):
        """Convert positions to Cartesian coordinates; no-op if already Cartesian."""
        if not self._impl.cartesian:
            self._impl.to_cartesian()

    # ── ASE interop ──────────────────────────────────────────────────

    def _site_energies_for_ase(self) -> Optional[np.ndarray]:
        """Return finite per-atom energies consistent with the stored total."""
        values = self.site_energies
        energy = self.energy
        if (values is None or values.shape != (len(self),)
                or energy is None or not np.isfinite(energy)
                or not np.all(np.isfinite(values))):
            return None
        if not np.isclose(values.sum(), energy, rtol=1e-10, atol=1e-10):
            return None
        return values

    def to_ase(self, attach_calculator: bool = True) -> "ase.Atoms":
        """Convert to ASE Atoms object.

        Requires ASE to be installed. By default, computed results are stored
        in a SinglePointCalculator and accessed through ASE ``get_*`` methods
        or ``calc.results``. Set ``attach_calculator=False`` to export the
        legacy ``info`` / ``arrays`` fields instead. Computed charges are kept
        separately from ASE initial charges.
        Computed magnetic moments similarly use ``magmoms`` results or legacy
        arrays, independently of ASE initial magnetic moments.
        Independent free energy uses ASE's ``free_energy`` result, or the
        same key in legacy ``info``, without replacing total energy.
        Available Cartesian velocities are converted from Angstrom/ps to
        ASE momenta, independently of ``attach_calculator``.
        Explicit per-atom masses are restored before converting velocities.
        Explicit tags, initial charges and initial magnetic moments are kept
        as input arrays, independently of computed results.
        Explicit total charge is input metadata in ``info['total_charge']``,
        independently of ``attach_calculator`` and computed atomic charges.
        """
        try:
            from ase import Atoms as ASEAtoms
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.units import fs
        except ImportError as exc:
            raise ImportError("ASE interop requires installing the optional 'ase' dependency") from exc

        numbers = self.numbers
        cartesian = self.copy()
        if not cartesian._impl.cartesian:
            cartesian.to_cartesian()
        positions = cartesian.positions
        cell = self.cell.copy()

        at = ASEAtoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=self.pbc,
        )
        if self.total_charge is not None:
            at.info['total_charge'] = self.total_charge
        masses = self.masses
        if masses is not None:
            at.set_masses(masses)
        tags = self.tags
        if tags is not None:
            at.set_tags(tags)
        initial_charges = self.initial_charges
        if initial_charges is not None:
            at.set_initial_charges(initial_charges)
        initial_magmoms = self.initial_magmoms
        if initial_magmoms is not None:
            at.set_initial_magnetic_moments(initial_magmoms)

        velocities = self.velocities
        if velocities is not None:
            if velocities.shape != (len(self), 3) or not np.all(np.isfinite(velocities)):
                raise ValueError("Velocities must have shape (N, 3) and contain only finite values")
            at.set_velocities(velocities / (1000 * fs))

        calc_results = {}

        # ASE writers require each result to have a single source.
        if self._impl.has_energy:
            if not attach_calculator:
                at.info['energy'] = self.energy
            calc_results["energy"] = float(self.energy)
        if self.free_energy is not None:
            if not attach_calculator:
                at.info['free_energy'] = self.free_energy
            calc_results['free_energy'] = self.free_energy
        site_energies = self._site_energies_for_ase()
        if site_energies is not None:
            if not attach_calculator:
                at.arrays['energies'] = site_energies.copy()
            calc_results["energies"] = site_energies
        charges = self.charges
        if charges is not None:
            if charges.shape != (len(self),) or not np.all(np.isfinite(charges)):
                raise ValueError("Computed charges must have shape (N,) and contain only finite values")
            if not attach_calculator:
                at.arrays['charges'] = charges.copy()
            calc_results['charges'] = charges
        magmoms = self.magmoms
        if magmoms is not None:
            if not np.all(np.isfinite(magmoms)):
                raise ValueError("Magnetic moments must be finite")
            if not attach_calculator:
                at.arrays['magmoms'] = magmoms.copy()
            calc_results['magmoms'] = magmoms
        if self._impl.has_forces:
            if not attach_calculator:
                at.arrays['forces'] = self.forces.copy()
            calc_results["forces"] = self.forces.copy()
        if self._impl.has_stress and self.stress is not None:
            s = self.stress
            stress_internal = np.array([
                s[0, 0], s[1, 1], s[2, 2],
                s[1, 2], s[0, 2], s[0, 1]
            ])
            volume = at.cell.volume
            if attach_calculator and volume > 0:
                calc_results["stress"] = -stress_internal / volume
            else:
                at.info['stress'] = stress_internal

        if attach_calculator and calc_results:
            at.calc = SinglePointCalculator(at, **calc_results)

        return at

    @staticmethod
    def from_ase(atoms: "ase.Atoms", read_results: bool = True) -> "Atoms":
        """Create imrmlp Atoms from ASE Atoms.

        Existing momenta supply Cartesian velocities in Angstrom/ps,
        independently of ``read_results``. Missing momenta stay absent.
        Explicit per-atom masses are retained independently of results.
        Explicit tags, initial charges and initial magnetic moments are also
        retained as input arrays; absent arrays stay absent.
        Explicit ``info['total_charge']`` is also input metadata and is retained
        independently of ``read_results``; atomic charges never supply it.
        Calculator results are imported only while valid for the current ASE
        state, without recalculation. Explicit legacy fields remain available;
        native EXTXYZ ``fenergy`` and ``site_en`` are accepted as fallbacks
        after standard ``free_energy`` and ``energies`` fields.

        Args:
            atoms: ASE Atoms object.
            read_results: Transfer available total/per-atom energy, free energy,
                forces, stress, computed charges and per-atom magnetic moments.
                Initial charges/magnetic moments are not imported as results.

        Returns:
            imrmlp.Atoms
        """
        numbers = atoms.get_atomic_numbers()
        imrmlp_at = Atoms(
            numbers=numbers,
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc().tolist(),
        )
        if 'total_charge' in atoms.info:
            imrmlp_at.total_charge = atoms.info['total_charge']
        if atoms.has('masses'):
            imrmlp_at.masses = np.array(atoms.get_masses(), dtype=np.float64, copy=True)
        if atoms.has('tags'):
            imrmlp_at.tags = atoms.get_tags()
        if atoms.has('initial_charges'):
            imrmlp_at.initial_charges = atoms.get_initial_charges()
        if atoms.has('initial_magmoms'):
            imrmlp_at.initial_magmoms = atoms.get_initial_magnetic_moments()

        if atoms.has('momenta'):
            from ase.units import fs
            velocities = np.array(atoms.get_velocities(), dtype=np.float64, copy=True) * (1000 * fs)
            if velocities.shape != (len(numbers), 3) or not np.all(np.isfinite(velocities)):
                raise ValueError("Velocities must have shape (N, 3) and contain only finite values")
            imrmlp_at._impl.velocities = velocities

        if not read_results:
            return imrmlp_at

        calculator = atoms.calc
        results = {}
        if calculator is not None and not calculator.check_state(atoms):
            results = calculator.results or {}

        # Transfer computed properties if present. calc.results follows ASE
        # stress units (eV/A^3); info['stress'] is kept as legacy internal virial.
        if "energy" in results:
            imrmlp_at._impl.energy = float(results["energy"])
        elif 'energy' in atoms.info:
            imrmlp_at._impl.energy = float(atoms.info['energy'])

        if "free_energy" in results:
            imrmlp_at.free_energy = results["free_energy"]
        elif 'free_energy' in atoms.info:
            imrmlp_at.free_energy = atoms.info['free_energy']
        elif 'fenergy' in atoms.info:
            imrmlp_at.free_energy = atoms.info['fenergy']

        if "energies" in results or "energies" in atoms.arrays or "site_en" in atoms.arrays:
            values = (results["energies"] if "energies" in results
                      else atoms.arrays["energies"] if "energies" in atoms.arrays
                      else atoms.arrays["site_en"])
            site_energies = np.array(values, dtype=np.float64, copy=True)
            if site_energies.shape != (len(numbers),):
                raise ValueError("ASE per-atom energies must have shape (N,)")
            imrmlp_at._impl.site_energies = site_energies

        if "charges" in results or "charges" in atoms.arrays:
            values = results["charges"] if "charges" in results else atoms.arrays["charges"]
            charges = np.array(values, dtype=np.float64, copy=True)
            if charges.shape != (len(numbers),) or not np.all(np.isfinite(charges)):
                raise ValueError("Computed charges must have shape (N,) and contain only finite values")
            imrmlp_at._impl.charges = charges

        if "magmoms" in results or "magmoms" in atoms.arrays:
            values = results["magmoms"] if "magmoms" in results else atoms.arrays["magmoms"]
            imrmlp_at.magmoms = values

        if "forces" in results:
            imrmlp_at._impl.forces = np.asarray(results["forces"], dtype=np.float64)
        elif 'forces' in atoms.arrays:
            imrmlp_at._impl.forces = atoms.arrays['forces']

        if "stress" in results:
            stress_voigt = np.asarray(results["stress"], dtype=np.float64)
            if stress_voigt.shape == (3, 3):
                s33 = stress_voigt
            else:
                s33 = np.array([
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
                ])
            imrmlp_at._impl.stress = -s33 * atoms.get_volume()
        elif 'stress' in atoms.info:
            stress_voigt = atoms.info['stress']
            # Convert Voigt [σxx, σyy, σzz, σyz, σxz, σxy] to 3x3 symmetric tensor
            s33 = np.array([
                [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
            ])
            imrmlp_at._impl.stress = s33

        return imrmlp_at

    def to_pymatgen(self):
        """Convert to Structure with per-axis PBC, or Molecule for no PBC.

        Molecule receives an explicit native total_charge when available;
        otherwise its default neutral charge is used. Structure charge and
        oxidation-state inference are outside this conversion's contract.
        """
        try:
            from pymatgen.core import Lattice, Molecule, Structure
        except ImportError as exc:
            raise ImportError("pymatgen interop requires installing the optional 'pymatgen' dependency") from exc

        species = [Z_TO_SYMBOL.get(int(z), str(int(z))) for z in self.numbers]
        cartesian = self.copy()
        if not cartesian._impl.cartesian:
            cartesian.to_cartesian()
        pbc = tuple(bool(flag) for flag in self.pbc)
        if any(pbc):
            return Structure(
                Lattice(self.cell, pbc=pbc),
                species,
                cartesian.positions,
                coords_are_cartesian=True,
                to_unit_cell=False,
            )
        if self.total_charge is None:
            return Molecule(species, cartesian.positions)
        return Molecule(species, cartesian.positions, charge=self.total_charge)

    @staticmethod
    def from_pymatgen(obj) -> "Atoms":
        """Create imrmlp Atoms from pymatgen Structure or Molecule.

        Molecule charge is retained as total_charge, including explicit zero.
        Molecule has no absent-charge state: exporting a native structure with
        no total_charge and importing it back therefore gives explicit zero.
        Structure charge and oxidation-state inference remain untransferred.
        """
        try:
            from pymatgen.core import Molecule, Structure
        except ImportError as exc:
            raise ImportError("pymatgen interop requires installing the optional 'pymatgen' dependency") from exc

        if isinstance(obj, Structure):
            numbers = np.array([site.specie.Z for site in obj], dtype=np.int32)
            return Atoms(
                numbers=numbers,
                positions=np.asarray(obj.cart_coords, dtype=np.float64),
                cell=np.asarray(obj.lattice.matrix, dtype=np.float64),
                pbc=obj.lattice.pbc,
            )
        if isinstance(obj, Molecule):
            numbers = np.array([site.specie.Z for site in obj], dtype=np.int32)
            positions = np.asarray(obj.cart_coords, dtype=np.float64)
            if len(positions):
                span = np.ptp(positions, axis=0)
                cell = np.diag(np.maximum(span + 20.0, 20.0))
            else:
                cell = np.eye(3) * 20.0
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=False)
            atoms.total_charge = obj.charge
            return atoms
        raise TypeError("Expected pymatgen Structure or Molecule")

    # ── Dunder methods ───────────────────────────────────────────────

    def __len__(self) -> int:
        return self.natoms

    def __repr__(self) -> str:
        n = self.natoms
        vols = f", vol={self.volume:.1f}" if n > 0 else ""
        engs = f", E={self.energy:.4f}" if self.energy is not None else ""
        uniq = sorted(set(int(z) for z in self.numbers))
        symbs = [Z_TO_SYMBOL.get(z, str(z)) for z in uniq]
        return f"Atoms(natoms={n}, species={symbs}{vols}{engs})"


def from_ase_list(structures: Sequence["ase.Atoms"]) -> List[Atoms]:
    """Convert a sequence of ASE Atoms objects to imrmlp Atoms."""
    return [Atoms.from_ase(s) for s in structures]


def to_ase_list(structures: Sequence[Atoms]) -> list:
    """Convert a sequence of imrmlp Atoms objects to ASE Atoms."""
    return [s.to_ase() for s in structures]


def from_pymatgen_list(structures: Sequence[object]) -> List[Atoms]:
    """Convert pymatgen Structure/Molecule objects to imrmlp Atoms."""
    return [Atoms.from_pymatgen(s) for s in structures]


def to_pymatgen_list(structures: Sequence[Atoms]) -> list:
    """Convert imrmlp Atoms objects to pymatgen structures."""
    return [s.to_pymatgen() for s in structures]
