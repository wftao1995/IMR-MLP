# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Nudged Elastic Band (NEB) — minimum energy path finding.

Wraps NEBDriver with a clean Python API.
Supports climbing-image NEB (CI-NEB) and variable-cell NEB (VC-NEB).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ._imrmlp import _NEB, _AnyPotential
from .atoms import Atoms


@dataclass
class NEBResult:
    """Result of NEB calculation.

    Attributes
    ----------
    converged : bool
        Whether NEB converged (max perpendicular force < ftol).
    n_iter : int
        Number of NEB iterations.
    images : List[Atoms]
        Relaxed images including endpoints.
    max_perp_force : float
        Maximum perpendicular force in the configured joint coordinate metric.
        For fixed Cartesian cells this has units eV/A.
    max_neb_force : float
        Maximum effective NEB force, including tangential spring/CI terms.
    stop_reason : str
        Structured termination reason such as ``converged`` or ``max_iterations``.
    """
    converged: bool
    n_iter: int
    images: List[Atoms]
    max_perp_force: float
    max_neb_force: float = 0.0
    max_spring_force: float = 0.0
    stop_reason: str = "not_run"
    n_force_evals: int = 0
    climbing_image: int = -1
    solver_policy: str = "legacy"
    accepted_steps: int = 0
    rejected_steps: int = 0
    n_image_force_evals: int = 0
    last_atomic_move: float = 0.0
    last_cell_change: float = 0.0
    last_fallback: str = "none"
    solver_phase: str = "ordinary"
    optimizer_iteration_units: str = "legacy_optimizer_iterations"
    force_evaluation_units: str = "band"
    symmetry_release_status: str = "disabled"
    climb_delay_units: str = "force_evaluations"


class NEB:
    """Nudged Elastic Band (NEB) method.

    Finds the minimum energy path between two endpoint structures.
    Fixed-cell Cartesian molecules need no padding cell. Fixed partial-periodic
    paths may omit open cell rows when the periodic rows are independent;
    minimum images preserve the open-direction motion. Variable-cell and fully
    periodic paths require a nonsingular cell.

    Parameters
    ----------
    potential : Potential
        imrmlp Potential for force evaluation.
    n_images : int
        Number of images including endpoints (default 7).
    spring : float
        Spring constant in eV/A^2 (default 5.0).
    climb : bool
        Enable climbing-image NEB (default False).
    climb_mode : str
        Select the highest-energy image or all local maxima (default 'highest').
    climb_delay : int
        Delay before enabling climbing image (default 0).
    vcneb : bool
        Variable-cell NEB (default False).
    pressure : float
        External pressure in GPa for VC-NEB (default 0).
    neb_ftol : float
        Force convergence tolerance in the configured joint metric (default 1e-2);
        eV/A for fixed Cartesian cells.
    tangent_type : int
        Tangent definition: 1=improved, 2=original (default 1).
        Stable-v1 safeguards every tangent using the complete force residual.
        A bounded matrix-free correction handles stalled force-flow steps;
        convergence does not by itself certify the saddle Hessian index.
    init_method : str
        Path initialization: 'linear' or 'idpp' (default 'linear').
    optimizer : str
        Optimizer: 'fire', 'lbfgs', 'sgd'/'gd', or 'cg' (default 'fire').
    max_iter : int
        Global optimizer iteration budget (default 1000).
    max_move : float
        Stable-v1: maximum actual displacement per atom in Angstrom, including
        affine cell motion (default 0.1). Legacy: native optimizer step bound.
    cell_weight : float
        VC-NEB cell-coordinate metric weight (default 100).
    solver_policy : str
        Numerical policy: 'legacy' (default pending material acceptance) or
        'stable-v1'. The physical path
        metric is unchanged. Climb delay counts accepted states in stable-v1.
    max_cell_strain : float
        Maximum spectral norm of the relative cell deformation minus identity
        per accepted step (default 0.002).
    max_force_evaluations : int, optional
        Stable-v1 band EFS budget, including rejected trials and Krylov probes.
        None or zero derives 10*max_iter+1. Endpoint EFS are counted separately
        in the result's n_image_force_evals.

    omp_mode : str
        Local OpenMP level: "images" (default) or "atoms". MPI workers are CLI-only.
    keep_symm : bool
        Constrain the common labelled path subgroup (full PBC required).
    symprec : float
        Symmetry detection tolerance in Angstrom (default 1e-5).
    symmetry_operations : str
        Comma-separated initial-image subgroup generators; requires keep_symm.
    symmetry_diagnostics : str
        Optional initial/final symmetry report filename.
    perturb_amplitude : float
        Maximum smooth initial atomic displacement in Angstrom; endpoints fixed.
        Incompatible with keep_symm. Defaults to zero.
    symmetry_release : str
        Optional new output directory for post-run symmetry release trials.
        Empty disables the diagnostic. The main path and counters are preserved.
        Only converged keep_symm paths are checked. Trials default to 200 steps,
        2001 band EFS, three seeds plus an unperturbed trial, atomic amplitude
        0.01 A, relative cell deformation 0.001 and barrier tolerance 1e-4 eV.
        Additional atomic displacement excludes affine motion due to cell strain.
        See report.json for skipped, incomplete or failed checks; no saddle or
        global minimality certification is implied.
    perturb_seed : int
        Nonnegative reproducible perturbation seed (default zero).

    Examples
    --------
    >>> neb = NEB(pot, n_images=7, climb=True)
    >>> result = neb.run(initial_atoms, final_atoms)
    >>> for i, img in enumerate(result.images):
    ...     print(f"Image {i}: {img.energy:.4f} eV")
    """

    def __init__(self,
                 potential,
                 n_images: int = 7,
                 spring: float = 5.0,
                 climb: bool = False,
                 climb_delay: int = 0,
                 vcneb: bool = False,
                 pressure: float = 0.0,
                 neb_ftol: float = 1e-2,
                 tangent_type: int = 1,
                 sproj_type: int = 1,
                 init_method: str = "linear",
                 check_mic: bool = True,
                 write_traj: bool = False,
                 nblock: int = 5,
                 optimizer: str = "fire",
                 max_iter: int = 1000,
                 max_move: float = 0.1,
                 variable_image: bool = False,
                 cell_weight: float = 100.0,
                 climb_mode: str = "highest",
                 *,
                 periodic_subspace: bool = False,
                 solver_policy: str = "legacy",
                 max_cell_strain: float = 0.002,
                 max_force_evaluations: Optional[int] = None,
                 omp_mode: str = "images",
                 keep_symm: bool = False,
                 symprec: float = 1e-5,
                 symmetry_operations: str = "",
                 symmetry_diagnostics: str = "",
                 perturb_amplitude: float = 0.0,
                 perturb_seed: int = 0,
                 symmetry_release: str = "",
                 symmetry_release_trials: int = 3,
                 symmetry_release_seed: int = 0,
                 symmetry_release_steps: int = 200,
                 symmetry_release_evaluations: int = 2001,
                 symmetry_release_amplitude: float = .01,
                 symmetry_release_strain: float = .001,
                 symmetry_release_energy_tol: float = 1e-4,
                 ):
        if n_images < 3:
            raise ValueError("n_images must include at least two endpoints and one interior image")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self._impl = _NEB()
        self._impl.omp_mode = omp_mode
        self._impl.keep_symm = keep_symm
        self._impl.symprec = symprec
        self._impl.symmetry_operations = symmetry_operations
        self._impl.symmetry_diagnostics = symmetry_diagnostics
        self._impl.perturb_amplitude = perturb_amplitude
        self._impl.perturb_seed = perturb_seed
        self._impl.symmetry_release = symmetry_release
        self._impl.symmetry_release_trials = symmetry_release_trials
        self._impl.symmetry_release_seed = symmetry_release_seed
        self._impl.symmetry_release_steps = symmetry_release_steps
        self._impl.symmetry_release_evaluations = symmetry_release_evaluations
        self._impl.symmetry_release_amplitude = symmetry_release_amplitude
        self._impl.symmetry_release_strain = symmetry_release_strain
        self._impl.symmetry_release_energy_tol = symmetry_release_energy_tol

        self._impl.solver_policy = solver_policy
        self._impl.max_cell_strain = max_cell_strain
        self._impl.max_force_evaluations = 0 if max_force_evaluations is None else max_force_evaluations
        self._n_images = n_images
        self._optimizer = optimizer
        self._max_iter = max_iter
        self._max_move = max_move
        self._impl.spring = spring
        self._impl.climb = climb
        self._impl.climb_mode = climb_mode
        self._impl.climb_delay = climb_delay
        self._impl.periodic_subspace = periodic_subspace
        self._impl.vcneb = vcneb
        self._impl.pressure = pressure
        self._impl.neb_ftol = neb_ftol
        self._impl.tangent_type = tangent_type
        self._impl.sproj_type = sproj_type
        self._impl.init_method = init_method
        self._impl.check_mic = check_mic
        self._impl.write_traj = write_traj
        if write_traj or symmetry_release:
            from ._imrmlp import _GPIO
            self._io = _GPIO()
            self._impl.iobj = self._io
        self._impl.nblock = nblock
        self._impl.variable_image = variable_image
        self._impl.cell_weight = cell_weight

        self._potential = potential
        self._pot_inner: _AnyPotential = (
            potential.inner if hasattr(potential, 'inner')
            else potential._impl.inner)

    def run(self, initial: Atoms, final: Atoms) -> NEBResult:
        """Run NEB between two endpoint structures.

        Args:
            initial: Starting structure.
            final: Ending structure.

        Returns:
            NEBResult with converged status and image chain.
        """
        # Leave interior images empty so the C++ coordinate codec performs one
        # consistent MIC-aware atomic/cell interpolation for fixed and VC NEB.
        n_interior = self._n_images - 2
        if initial._elements != final._elements:
            raise ValueError("NEB endpoints must use the same element mapping")
        images = [initial.copy()]
        images.extend(Atoms() for _ in range(n_interior))
        charge = initial.total_charge
        if charge is not None and charge == final.total_charge:
            for image in images[1:]:
                image.total_charge = charge
        images.append(final.copy())

        # Convert to C++ Atoms list
        cpp_images = [img._impl for img in images]

        # Initialize NEB
        self._impl.pot = self._pot_inner
        if self._impl.write_traj or self._impl.symmetry_release:
            self._io.set_elements(initial._elements or [])
        self._impl.init(cpp_images)

        # Run
        self._impl.run(self._optimizer, self._max_iter, self._max_move)

        # Collect results (deep copy to avoid use-after-free if NEB driver is GC'd)
        result_images = [Atoms(_impl=a.copy(), elements=initial._elements)
                         for a in self._impl.images]
        result = NEBResult(
            converged=self._impl.converged,
            symmetry_release_status=self._impl.symmetry_release_status,
            n_iter=self._impl.optimizer_iterations,
            images=result_images,
            max_perp_force=self._impl.max_perp_force,
            max_neb_force=self._impl.max_neb_force,
            max_spring_force=self._impl.max_spring_force,
            stop_reason=self._impl.stop_reason,
            n_force_evals=self._impl.force_evaluations,
            climbing_image=self._impl.climbing_image,
            solver_policy=self._impl.solver_policy,
            accepted_steps=self._impl.accepted_steps,
            rejected_steps=self._impl.rejected_steps,
            n_image_force_evals=self._impl.image_force_evaluations,
            last_atomic_move=self._impl.last_atomic_move,
            last_cell_change=self._impl.last_cell_change,
            last_fallback=self._impl.last_fallback,
            solver_phase=self._impl.solver_phase,
            optimizer_iteration_units=("attempts" if self._impl.solver_policy == "stable-v1"
                                       else "legacy_optimizer_iterations"),
            climb_delay_units=("accepted_steps" if self._impl.solver_policy == "stable-v1"
                               else "force_evaluations"),
        )
        return result

    def __repr__(self) -> str:
        return (f"NEB(n_images={self._n_images}, "
                f"spring={self._impl.spring}, climb={self._impl.climb})")


def make_path_seed(images, *, settings=None, reference_cell=None, mode="sine1",
                   seed=0, sign=1, amplitude=0.01, strain=0.001,
                   atoms=None, center=None, radius=0.0):
    """Generate a bounded complete-path seed without a potential or EFS calls.

    ``settings`` uses unprefixed native NEB names (e.g. ``vcneb``,
    ``recast_atomic_force`` and ``ciival``). VC paths require their original
    reference cell. Images must already be ordered, complete and unfolded.
    ``atoms`` and ``center`` use zero-based original atom labels; no reordering
    or automatic matching is performed. Returns (new_images, diagnostics).
    """
    import numpy as np
    from ._imrmlp import _neb_path_seed
    if len(images) < 3:
        raise ValueError("At least three complete images are required")
    if any(image._elements != images[0]._elements for image in images):
        raise ValueError("All images must use the same element mapping")
    config = {k: str(v).lower() if isinstance(v, bool) else str(v)
              for k, v in (settings or {}).items()}
    if config.get("vcneb", "false") == "true" and reference_cell is None:
        raise ValueError("VC path seeds require the original reference_cell")
    reference = np.eye(3) if reference_cell is None else np.asarray(reference_cell, dtype=float)
    native, info = _neb_path_seed([image._impl for image in images], config,
        reference, mode, seed, sign, amplitude, strain, [] if atoms is None else list(atoms),
        -1 if center is None else center, radius)
    return [Atoms(_impl=a.copy(), elements=images[0]._elements) for a in native], dict(info)
