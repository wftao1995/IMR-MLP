"""Explicit numeric conversions and declared-source structure normalization.

These helpers do not infer units from labels. Canonical Atoms units are
Angstrom, eV, ps and amu; the stored ``stress`` tensor is an energy virial.
"""
from collections.abc import Mapping
import numpy as np
from . import _imrmlp
from .atoms import Atoms

__all__ = ["unit_conversion_factor", "convert_units", "source_unit_system",
           "canonicalize_atoms", "pressure_to_virial_factor"]

unit_conversion_factor = _imrmlp.unit_conversion_factor
source_unit_system = _imrmlp.source_unit_system
pressure_to_virial_factor = _imrmlp.pressure_to_virial_factor


def convert_units(value, quantity, from_unit, to_unit):
    """Convert a scalar or numeric array using an explicit physical quantity."""
    factor = unit_conversion_factor(quantity, from_unit, to_unit)
    result = np.asarray(value) * factor
    return result.item() if result.ndim == 0 else result


def _source_units(source):
    if isinstance(source, str):
        units = dict(source_unit_system(source))
    elif isinstance(source, Mapping):
        units = dict(source_unit_system("canonical"))
        unknown = set(source) - set(units)
        if unknown:
            raise ValueError(f"Unknown source unit fields: {sorted(unknown)}")
        units.update(source)
    else:
        raise TypeError("input_units must be a preset string or a base-unit mapping")
    # Validate even an empty import, before opening input or producing output.
    for quantity, target in (("length", "A"), ("energy", "eV"),
                             ("time", "ps"), ("mass", "amu")):
        unit_conversion_factor(quantity, units[quantity], target)
    return units


def canonicalize_atoms(atoms, source="canonical", **base_units):
    """Return a normalized copy, preserving the explicit element mapping.

    ``source`` is ``canonical``, ``metal``, ``atomic``, ``real`` or a mapping of
    length/energy/time/mass units. Keyword base units override that declaration.
    Fractional positions, arbitrary attributes and dimensionless HVP probes are
    retained. Known energies, virial, forces, velocities, masses and curvature
    labels use the declared base units; missing labels remain missing.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be a imrmlp.Atoms")
    units = _source_units(source)
    units.update(base_units)
    units = _source_units(units)
    return Atoms(_impl=_imrmlp.canonicalize_atoms(atoms._impl, **units),
                 elements=atoms._elements)
