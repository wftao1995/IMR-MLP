# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Active Learning module — selection algorithms and uncertainty estimation.

Selection algorithms:
  - ActiveSelector  — MaxVol determinant-based selection + extrapolation grading
  - FPSSelector     — Farthest Point Sampling diversity selection
  - cur_select()    — CUR leverage-score selection

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from ._imrmlp import (
    _ActiveSelector, _FPSSelector,
    _AnyPotential, _MaxVol, _build_bmatrix, cur_select_rows,
)
from .atoms import Atoms
from .utils import _coerce_atoms


# ============================================================================
#  ActiveSelector — MaxVol-based selection
# ============================================================================

class MaxVol:
    """Matrix-level Maximum Volume row selection.

    This low-level API operates directly on NumPy matrices. For structure
    pools, use build_bmatrix() or ActiveSelector.
    """

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100,
                 max_swaps: int = 0, verbose: bool = False):
        self._impl = _MaxVol()
        self._impl.tolerance = tolerance
        self._impl.max_iterations = max_iterations
        self._impl.max_swaps = max_swaps
        self._impl.verbose = verbose

    def select_rows(self, A: np.ndarray, k: int) -> List[int]:
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A must be a 2D matrix")
        return list(self._impl.select_rows(A, int(k)))

    def compute_scores(self, A: np.ndarray, selected: List[int]) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        return np.asarray(self._impl.compute_scores(A, list(selected)), dtype=np.float64)

    def add_rows(self, A: np.ndarray, current: List[int], add_k: int) -> List[int]:
        A = np.asarray(A, dtype=np.float64)
        return list(self._impl.add_rows(A, list(current), int(add_k)))

    @property
    def log_determinant(self) -> float:
        return float(self._impl.log_determinant)

    @property
    def swap_history(self):
        return list(self._impl.swap_history)


def build_bmatrix(structures: List[object],
                  potfile: str,
                  pottype: str = "mtp",
                  dtype: str = "mtm",
                  mode: str = "nbh",
                  params: Optional[Dict[str, str]] = None,
                  weight: float = 1.0,
                  pow_scale: float = 0.5,
                  w_energy: float = 1.0,
                  w_site: float = 0.0,
                  w_force: float = 0.0,
                  w_stress: float = 0.0,
                  all_coeffs: bool = False) -> np.ndarray:
    """Build the MaxVol B-matrix for imrmlp/ASE/pymatgen structures."""
    cpp_atoms = [_coerce_atoms(s)._impl for s in structures]
    return np.asarray(_build_bmatrix(
        cpp_atoms, str(potfile), pottype, dtype, mode, params or {},
        weight, pow_scale, w_energy, w_site, w_force, w_stress, all_coeffs,
    ), dtype=np.float64)


class ActiveSelector:
    """MaxVol-based active set selection and extrapolation grading.

    Manages a pool of candidate structures, builds B-matrix rows,
    runs MaxVol selection, and grades new structures.

    Parameters
    ----------
    threshold : float
        MaxVol selection threshold (default 1.001).
    weight : float
        B-matrix row weight multiplier (default 1.0).
    pow_scale : float
        natoms scaling exponent (default 0.5).
    max_swaps : int
        Max Vol swaps per iteration (0 = unlimited, default 0).

    Examples
    --------
    >>> aselect = ActiveSelector(threshold=1.001)
    >>> aselect.add_candidates(pool_structures)
    >>> aselect.select()
    >>> grade = aselect.grade(new_structure)

    Persistence:
    >>> aselect.save("active_set.sel")
    >>> aselect2 = ActiveSelector.load("active_set.sel")
    """

    def __init__(self,
                 threshold: float = 1.001,
                 weight: float = 1.0,
                 pow_scale: float = 0.5,
                 max_swaps: int = 0,
                 potential=None,
                 potfile: Optional[str] = None,
                 pottype: str = "mtp",
                 dtype: str = "mtm",
                 mode: str = "nbh",
                 params: Optional[Dict[str, str]] = None,
                 w_energy: float = 1.0,
                 w_site: float = 0.0,
                 w_force: float = 0.0,
                 w_stress: float = 0.0,
                 all_coeffs: bool = False,
                 select_mode: str = "stream",
                 stream_mode: str = "exact",
                 stream_chunk_size: int = 32,
                 stream_fast_passes: int = 2,
                 ):
        if potfile is not None:
            self._impl = _ActiveSelector(
                str(potfile), pottype, dtype, mode, params or {},
                w_energy, w_site, w_force, w_stress, all_coeffs,
            )
        else:
            self._impl = _ActiveSelector()
            if potential is not None:
                pot_inner = potential.inner if hasattr(potential, 'inner') else potential
                self._impl.init_builder(pot_inner)
        self._impl.threshold = threshold
        self._impl.weight = weight
        self._impl.pow_scale = pow_scale
        self._impl.max_swaps = max_swaps
        self._impl.select_mode = select_mode
        self._impl.stream_mode = stream_mode
        self._impl.stream_chunk_size = stream_chunk_size
        self._impl.stream_fast_passes = stream_fast_passes

    def add_candidates(self, structures: List[object]) -> int:
        """Add candidate structures to the B-matrix pool."""
        cpp_atoms = [_coerce_atoms(s)._impl for s in structures]
        return self._impl.add_candidates(cpp_atoms)

    def clear_candidates(self):
        """Clear the candidate pool."""
        self._impl.clear_candidates()

    @property
    def n_candidates(self) -> int:
        return self._impl.num_candidates()

    @property
    def n_rows(self) -> int:
        return self._impl.num_rows()

    def select(self, threshold: Optional[float] = None) -> int:
        """Run MaxVol selection."""
        if threshold is not None:
            self._impl.threshold = threshold
        return self._impl.select(self._impl.threshold)

    def grade(self, structure: Atoms) -> float:
        """Compute MaxVol extrapolation grade.

        Returns grade >= 1.0. 1.0 = inside active set,
        1-10 = mild extrapolation, >>10 = extreme extrapolation.
        """
        return self._impl.grade(structure._impl)

    def grade_all(self, structures: List[object]) -> np.ndarray:
        """Batch grade structures."""
        cpp_atoms = [_coerce_atoms(s)._impl for s in structures]
        return np.asarray(self._impl.grade_all(cpp_atoms), dtype=np.float64)

    @property
    def selected_indices(self) -> List[int]:
        return list(self._impl.selected_indices)

    @property
    def selected_structures(self) -> List[Atoms]:
        return [Atoms(_impl=a) for a in self._impl.get_selected()]

    def save(self, filename: str):
        """Save a legacy .sel state or a self-contained .gpm bundle."""
        self._impl.save_state(str(filename), "", "", "")

    @staticmethod
    def load(filename: str, potential=None):
        """Load a legacy .sel or the bound MaxVol state in a GPM bundle."""
        if str(filename).endswith(".gpm"):
            from .model_bundle import ModelBundle
            bundle = ModelBundle.load(filename)
            if potential is not None:
                raise ValueError("A GPM selector uses its embedded potential; load .sel to bind an external model")
            return bundle.create_al().selector
        aselect = ActiveSelector.__new__(ActiveSelector)
        aselect._impl = _ActiveSelector.load_state(str(filename), "")
        if potential is not None:
            pot_inner = potential.inner if hasattr(potential, 'inner') else potential._impl.inner
            aselect._impl.init_builder(pot_inner)
        return aselect


# ============================================================================
#  FPSSelector — Farthest Point Sampling
# ============================================================================

class FPSSelector:
    """Farthest Point Sampling for diversity selection.

    Parameters
    ----------
    metric : str
        Distance metric: 'euclidean' or 'cosine'.
    random_seed : int
        Random seed for reproducible starts (default 42).

    Examples
    --------
    >>> fps = FPSSelector()
    >>> fps.add_candidates(pool)
    >>> selected = fps.select(n=50)
    """

    def __init__(self, metric: str = "euclidean", random_seed: int = 42):
        self._impl = _FPSSelector()
        # FPSSelector doesn't have metric/seed as member fields exposed —
        # they're set via BMatrixBuilder
        self._metric = metric
        self._seed = random_seed

    def add_candidates(self, structures: List[Atoms]):
        cpp_atoms = [s._impl for s in structures]
        self._impl.add_candidates(cpp_atoms)

    def select(self, n: int, first_idx: int = -1) -> List[Atoms]:
        self._impl.select(n, first_idx)
        return [Atoms(_impl=a) for a in self._impl.get_selected()]

    def select_by_threshold(self, dist_thr: float, max_n: int = 0,
                            first_idx: int = -1):
        self._impl.select_by_threshold(dist_thr, max_n, first_idx)

    @property
    def selected_indices(self) -> List[int]:
        return list(self._impl.selected_indices)

    @property
    def selection_scores(self) -> np.ndarray:
        return np.array(self._impl.selection_scores)


# ============================================================================
#  CUR selection
# ============================================================================

@dataclass
class CURResult:
    selected_indices: List[int]
    scores: np.ndarray
    reconstruction_error: float


def cur_select(A: np.ndarray, k: int,
               deterministic: bool = False,
               seed: int = 42,
               compute_error: bool = False) -> CURResult:
    """CUR leverage-score row selection from a descriptor matrix.

    Args:
        A: (n_structures, feature_dim) descriptor matrix.
        k: Number of structures to select.
        deterministic: If True, top-k by score; if False, probability-weighted.
        seed: Random seed for probabilistic selection.
        compute_error: Compute relative reconstruction error.
    """
    result = cur_select_rows(A, k, deterministic, seed, compute_error)
    return CURResult(
        selected_indices=result["selected_indices"],
        scores=np.array(result["scores"]),
        reconstruction_error=result.get("reconstruction_error", -1.0),
    )


def select_from_pool(structures: List[object], potfile: str, **kwargs) -> ActiveSelector:
    """Build a MaxVol active selector from a structure pool."""
    selector_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs)
        if key in {
            "threshold", "weight", "pow_scale", "max_swaps", "pottype",
            "dtype", "mode", "params", "w_energy", "w_site", "w_force",
            "w_stress", "all_coeffs",
        }
    }
    selector = ActiveSelector(potfile=potfile, **selector_kwargs)
    selector.add_candidates(structures)
    selector.select()
    return selector
