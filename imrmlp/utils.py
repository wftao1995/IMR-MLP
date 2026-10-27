# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Utility functions: element mapping, unit conversions."""

from typing import Dict, List, Optional
import numpy as np


_CHEMICAL_SYMBOLS = [
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# Periodic table: atomic_number -> symbol
Z_TO_SYMBOL: Dict[int, str] = {
    z: symb for z, symb in enumerate(_CHEMICAL_SYMBOLS) if symb != "X"
}

# Symbol -> atomic_number
SYMBOL_TO_Z: Dict[str, int] = {symb: z for z, symb in Z_TO_SYMBOL.items()}


def build_type_mapping(elements: List[str]) -> Dict[int, int]:
    """Build atomic_number -> chemmap_index mapping from element list.

    Args:
        elements: List of element symbols or atomic number strings,
                  e.g. ['Al', 'Mg'] or ['13', '12']

    Returns:
        Dict mapping atomic number to 0-based species index.

    Example:
        >>> build_type_mapping(['Al', 'Mg'])
        {13: 0, 12: 1}
    """
    result = {}
    for i, el in enumerate(elements):
        # Try as element symbol first
        z = SYMBOL_TO_Z.get(el)
        if z is None:
            # Try parsing as atomic number string
            try:
                z = int(el)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot interpret element '{el}': "
                                 "not a valid element symbol or atomic number")
        result[z] = i
    return result


def build_inverse_mapping(elements: List[str]) -> Dict[int, str]:
    """Build chemmap_index -> element_symbol mapping.

    Example:
        >>> build_inverse_mapping(['Al', 'Mg'])
        {0: 'Al', 1: 'Mg'}
    """
    return {i: el for i, el in enumerate(elements)}


def numbers_from_chemmap(chemmap: list, index_to_symbol: Dict[int, str]) -> np.ndarray:
    """Convert chemmap indices to atomic numbers.

    Args:
        chemmap: Species index per atom (from Atoms.chemmap)
        index_to_symbol: Mapping from chemmap index to element symbol or atomic number string.

    Returns:
        numpy array of atomic numbers.
    """
    numbers = []
    for t in chemmap:
        element = index_to_symbol.get(t)
        z = SYMBOL_TO_Z.get(element)
        if z is None:
            try:
                z = int(element)
            except (ValueError, TypeError):
                z = t
        numbers.append(z)
    return np.array(numbers, dtype=np.int32)


def chemmap_from_numbers(numbers: np.ndarray, elements: Optional[List[str]] = None) -> list:
    """Build chemmap from atomic numbers.

    If elements is None, auto-builds mapping from unique values in numbers.
    An explicit element list must contain every input atomic number.

    Args:
        numbers: Atomic numbers per atom.
        elements: Element symbols list, in order of chemmap indices.

    Returns:
        List of integer chemmap indices.
    """
    if elements is not None:
        type_map = build_type_mapping(elements)
        try:
            return [type_map[int(n)] for n in numbers]
        except KeyError as exc:
            raise ValueError(f"Atomic number {exc.args[0]} is not present in elements") from None

    # Auto-build mapping from unique species
    uniq = sorted(np.unique(numbers))
    type_map = {int(z): i for i, z in enumerate(uniq)}
    return [type_map[int(n)] for n in numbers]


def _coerce_atoms(structure: object, *, copy_atoms: bool = False):
    from .atoms import Atoms

    if isinstance(structure, Atoms):
        return structure.copy() if copy_atoms else structure
    module = type(structure).__module__
    if module.startswith("ase.") or (
        hasattr(structure, "get_atomic_numbers") and hasattr(structure, "get_positions")
    ):
        return Atoms.from_ase(structure)
    if module.startswith("pymatgen.") or (
        hasattr(structure, "cart_coords") and hasattr(structure, "sites")
    ):
        return Atoms.from_pymatgen(structure)
    raise TypeError("Expected imrmlp.Atoms, ASE Atoms, or pymatgen Structure/Molecule")
