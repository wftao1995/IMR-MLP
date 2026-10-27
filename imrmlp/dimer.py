# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Dimer method — saddle point (transition state) search.

Wraps DimerDriver with a clean Python API.
"""

from __future__ import annotations
from dataclasses import dataclass

from ._imrmlp import _Dimer, _AnyPotential
from .atoms import Atoms


@dataclass
class DimerResult:
    """Result of a dimer saddle-point search.

    Attributes
    ----------
    converged : bool
        Whether the dimer converged.
    n_iter : int
        Number of outer iterations.
    energy : float
        Center energy in eV (enthalpy E+pV for variable-cell searches).
    curvature : float
        Curvature along the dimer mode; a negative value alone does not prove saddle order.
    saddle_atoms : Atoms
        Structure at the saddle point.
    """
    converged: bool
    n_iter: int
    energy: float
    curvature: float
    saddle_atoms: Atoms
    exit_code: int = 1
    stop_reason: str = "not_run"
    max_force: float = 0.0
    max_rot_gradient: float = 0.0


class Dimer:
    """Dimer method for saddle point search.

    Creates a dimer (two images separated by ddr along a mode direction)
    and iteratively rotates to find the lowest curvature mode and translates
    uphill toward the saddle point.

    Fixed-cell searches accept Cartesian molecules without a cell and slabs
    with independent periodic cell rows and missing open rows. Variable-cell,
    fully periodic and fractional inputs require a nonsingular cell.

    Parameters
    ----------
    potential : Potential
        imrmlp Potential for force evaluation.
    ddr : float
        Dimer separation distance in A (default 5e-3).
    max_iter : int
        Maximum outer iterations (default 1000).
    dimer_gtol : float
        Force convergence tolerance in eV/A (default 1e-2).
    rot_driver : str
        Rotation sub-optimizer: 'fire', 'lbfgs', 'cg' (default 'fire').
    trn_driver : str
        Translation sub-optimizer: 'fire', 'lbfgs', 'cg' (default 'fire').
    rot_max_iter : int
        Max rotation iterations per outer step (default 50).
    trn_max_iter : int
        Max translation iterations per outer step (default 200).
    vc_flag : bool
        Variable-cell dimer (default False).
    pressure : float
        External pressure in GPa for VC (default 0).
    write_traj : bool
        Write trajectory file (default False).

    modecar : str
        Explicit MODECAR path; invalid files raise an exception.
    random_seed : int
        Random direction seed if modecar is empty (0 by default; -1 for nondeterministic).
    rot_gtol, trn_gtol : float
        Rotation curvature-gradient and translation-gradient tolerances.
    symm_strain, restore_F_cholesky, recast_atomic_force : bool
        VC coordinate convention; must match a versioned MODECAR header.

    Examples
    --------
    >>> dimer = Dimer(pot, ddr=5e-3, max_iter=500)
    >>> result = dimer.run(atoms)
    >>> print(result.converged, result.energy, result.curvature)
    """

    def __init__(self,
                 potential,
                 ddr: float = 5e-3,
                 max_iter: int = 1000,
                 dimer_gtol: float = 1e-2,
                 rot_driver: str = "fire",
                 trn_driver: str = "fire",
                 rot_max_iter: int = 50,
                 trn_max_iter: int = 200,
                 vc_flag: bool = False,
                 pressure: float = 0.0,
                 write_traj: bool = False,
                 *,
                 modecar: str = "",
                 random_seed: int = 0,
                 rot_gtol: float = 1e-3,
                 trn_gtol: float = 1e-2,
                 symm_strain: bool = True,
                 restore_F_cholesky: bool = True,
                 recast_atomic_force: bool = True,
                 periodic_subspace: bool = False,
                 ):
        self._impl = _Dimer()
        self._impl.modecarf = str(modecar)
        self._impl.random_seed = random_seed
        self._impl.rot_gtol = rot_gtol
        self._impl.trn_gtol = trn_gtol
        self._impl.symm_strain = symm_strain
        self._impl.restore_F_cholesky = restore_F_cholesky
        self._impl.recast_atomic_force = recast_atomic_force
        self._impl.ddr = ddr
        self._impl.max_iter = max_iter
        self._impl.dimer_gtol = dimer_gtol
        self._impl.rot_driver = rot_driver
        self._impl.trn_driver = trn_driver
        self._impl.rot_max_iter = rot_max_iter
        self._impl.trn_max_iter = trn_max_iter
        self._impl.periodic_subspace = periodic_subspace
        self._impl.vc_flag = vc_flag
        self._impl.pressure = pressure
        self._impl.write_traj = write_traj
        if write_traj:
            from ._imrmlp import _GPIO
            self._io = _GPIO()
            self._impl.iobj = self._io

        self._potential = potential
        self._pot_inner: _AnyPotential = (
            potential.inner if hasattr(potential, 'inner')
            else potential._impl.inner)

    def run(self, atoms: Atoms) -> DimerResult:
        """Run dimer search from the given structure.

        Args:
            atoms: Initial structure near the saddle point.

        Returns:
            DimerResult with convergence status and saddle structure.
        """
        self._impl.pot = self._pot_inner
        if self._impl.write_traj:
            self._io.set_elements(atoms._elements or [])
        self._impl.init(atoms._impl)

        exit_code = self._impl.run()

        saddle = Atoms(_impl=self._impl.atom_center.copy(), elements=atoms._elements)
        return DimerResult(
            converged=self._impl.converged,
            n_iter=self._impl.n_iter,
            energy=self._impl.energy,
            curvature=self._impl.curvature,
            saddle_atoms=saddle,
            exit_code=exit_code,
            stop_reason="converged" if self._impl.converged else "max_iterations",
            max_force=self._impl.max_center_gradient,
            max_rot_gradient=self._impl.max_rot_force,
        )

    def __repr__(self) -> str:
        return (f"Dimer(ddr={self._impl.ddr}, "
                f"max_iter={self._impl.max_iter}, "
                f"dimer_gtol={self._impl.dimer_gtol})")
