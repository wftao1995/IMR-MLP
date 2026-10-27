# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Structure relaxation driver.

Wraps RelaxationDrvV2 with a clean Python API.
Supports L-BFGS, FIRE, CG, GD, MDMin and QuickMin optimizers.
Variable-cell relaxation and constraints are supported.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import json
import numpy as np

from ._imrmlp import _Relaxation, _AnyPotential
from .atoms import Atoms


# Optimizer constants
LBFGS = "lbfgs"
FIRE = "fire"
CG = "cg"
GD = "gd"
MDMIN = "mdmin"
QUICKMIN = "quickmin"


@dataclass
class RelaxResult:
    """Result of a structure relaxation.

    Attributes
    ----------
    converged : bool
        Whether the relaxation converged.
    n_iter : int
        Number of iterations.
    final_energy : float
        Final optimization objective in eV, including pressure and AUL terms.
        The potential energy is available as final_atoms.energy.
    init_energy : float
        Initial energy in eV.
    final_atoms : Atoms
        Relaxed structure.
    """
    converged: bool
    n_iter: int
    final_energy: float
    init_energy: float
    final_atoms: Atoms
    stop_reason: str = "not_started"


class Relaxation:
    """Structure relaxation driver.

    Fixed-cell default recast coordinates support Cartesian molecules without
    a cell and slabs with independent periodic rows and missing open rows.
    Variable-cell and fractional coordinate modes retain their cell requirements.

    Parameters
    ----------
    potential : Potential
        imrmlp Potential object for force evaluation.
    driver : str
        Optimizer: 'lbfgs', 'lbfgspp', 'fire', 'cg', 'gd', 'mdmin', 'quickmin', 'sqp' or 'nlc'.
    timestep : float, optional
        Optimization timestep for MDMin or QuickMin (native default 0.2).
        These are unit-mass search dynamics in the active coordinates.
    max_iter : int
        Maximum iterations (default 1000).
    gtol : float
        Gradient component tolerance in active optimizer coordinates (default
        1e-3). Cartesian atomic coordinates give force units of eV/A.
    ftol : float
        Energy convergence tolerance in eV (default 0).
    xtol : float
        Step tolerance in active optimizer coordinates (default 0).
    relax_cell : bool
        Allow cell relaxation (default False).
    fix_vol : bool
        Fix cell volume during relaxation (default False).
    fix_shape : bool
        Fix cell shape during relaxation (default False).
    pressure : float
        External pressure in GPa (default 0).
    max_move : float
        Maximum step norm in active optimizer coordinates (default 0.1).
        Cartesian atomic coordinates give displacement units of Angstrom.
    trajfile : str
        If set, write relaxation trajectory to this file.
    downhill_check : bool
        With FIRE, reject steps that increase the complete objective. Rejected
        trials count toward max_iter; the default is False.
    constraints : list of dict, optional
        Version-1 constraint entries, including distance and angle. Atom indices
        are zero-based; angle i-j-k has vertex j. Targets use Angstrom or degrees.
        Omitted targets bind to each run's starting structure. Periodic bond
        images are selected initially and held fixed.
    constraint_version : int
        JSON schema version (1 preserves existing equalities; 2 enables bounds,
        walls, formulas, dihedrals and selected symmetry operations).
    constraint_loops : int
        Maximum augmented-Lagrangian outer iterations (default 10).
    force_only : bool
        With L-BFGS++, choose steps from directional gradients and disable its
        energy-change stopping test. Energy and forces are still evaluated.
    max_linesearch : int
        Maximum L-BFGS++ line-search trials (default 20). Tight constrained
        optimizations may need a larger value, such as 80.

    Examples
    --------
    >>> from imrmlp import Potential, read_one, Relaxation
    >>> pot = Potential("Al.mtp")
    >>> atoms = read_one("POSCAR")
    >>> relax = Relaxation(pot, driver="fire", gtol=1e-2, relax_cell=True)
    >>> result = relax.run(atoms)
    >>> print(result.converged, result.final_energy)
    """

    def __init__(self,
                 potential,
                 driver: str = "lbfgs",
                 max_iter: int = 1000,
                 gtol: float = 1e-3,
                 ftol: float = 0.0,
                 xtol: float = 0.0,
                 relax_cell: bool = False,
                 fix_vol: bool = False,
                 fix_shape: bool = False,
                 pressure: float = 0.0,
                 max_move: float = 0.1,
                 selective_dynamics: bool = False,
                 trajfile: str = "",
                 downhill_check: bool = False,
                 constraints: Optional[List[dict]] = None,
                 constraint_loops: int = 10,
                 max_linesearch: int = 20,
                 timestep: Optional[float] = None,
                 force_only: bool = False,
                 constraint_version: int = 1,
                 periodic_subspace: bool = False,
                 hessian_precond_file: str = "",
                 sparse_precond: bool = False,
                 sparse_shift: float = 1e-3,
                 precond_floor: float = 1e-8,
                 fc_update_interval: int = 0,
                 aul_auto: bool = False,
                 thin_cell_scaling: bool = False,
                 rigid_groups=None,
                 coordinate_map: Optional[dict] = None,
                 ):
        self._impl = _Relaxation()

        # Set parameters
        self._impl.driver = driver
        self._impl.periodic_subspace = periodic_subspace
        self._impl.hessian_precond_file = hessian_precond_file
        self._impl.sparse_precond = sparse_precond
        self._impl.sparse_shift = sparse_shift
        self._impl.precond_floor = precond_floor
        self._impl.fc_update_interval = fc_update_interval
        self._impl.aul_auto = aul_auto
        self._impl.thin_cell_scaling = thin_cell_scaling
        if rigid_groups is not None:
            self._impl.rigid_groups = (rigid_groups if isinstance(rigid_groups, str) else
                ";".join(",".join(str(i) for i in group) for group in rigid_groups))
        if coordinate_map is not None:
            self._impl.set_coordinate_map(**coordinate_map)
        if force_only:
            if driver != "lbfgspp":
                raise ValueError("force_only requires the lbfgspp optimizer")
            self._impl.force_only = True
        if timestep is not None:
            if driver == MDMIN:
                self._impl.mdmin_dt = float(timestep)
            elif driver == QUICKMIN:
                self._impl.quickmin_dt = float(timestep)
            else:
                raise ValueError("timestep applies to mdmin or quickmin")
        self._impl.max_iter = max_iter
        self._impl.gtol = gtol
        self._impl.ftol = ftol
        self._impl.xtol = xtol
        self._impl.relax_cell = relax_cell
        self._impl.fix_vol = fix_vol
        self._impl.fix_shape = fix_shape
        self._impl.pressure = pressure
        self._impl.max_move = max_move
        self._impl.selective_dynamics = selective_dynamics
        self._impl.downhill_check = downhill_check
        self._impl.constraint_loops = constraint_loops
        self._impl.max_linesearch = max_linesearch
        self._constraints_json = (json.dumps({"version": constraint_version, "constraints": constraints})
                                  if constraints else "")
        self._impl.write_traj = bool(trajfile)
        if trajfile:
            self._impl.trajfile = trajfile

        self._potential = potential
        self._pot_inner: _AnyPotential = (
            potential.inner if hasattr(potential, 'inner')
            else potential._impl.inner)

    def run(self, atoms: Atoms) -> RelaxResult:
        """Run relaxation starting from the given atomic structure.

        Args:
            atoms: Starting structure (modified with relaxed positions).

        Returns:
            RelaxResult with convergence status and relaxed structure.
        """
        # Init with potential and a copy of the atoms
        self._impl.set_elements(atoms._elements or [])
        self._impl.init(self._pot_inner, atoms._impl, self._constraints_json)

        # Run
        self._impl.run()
        atoms._impl = self._impl.atom.copy()

        return RelaxResult(
            converged=self._impl.converged,
            n_iter=self._impl.n_iter,
            final_energy=self._impl.energy,
            init_energy=self._impl.energy0,
            final_atoms=atoms,
            stop_reason=self._impl.stop_reason,
        )

    @property
    def converged(self) -> bool:
        """Whether the last run converged."""
        return self._impl.converged

    @property
    def stop_reason(self):
        return self._impl.stop_reason

    @property
    def energy(self) -> float:
        """Final objective, including pressure and AUL terms when present."""
        return self._impl.energy

    def __repr__(self) -> str:
        return (f"Relaxation(driver='{self._impl.driver}', "
                f"max_iter={int(self._impl.max_iter)}, "
                f"gtol={self._impl.gtol})")
