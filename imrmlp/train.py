# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Linear fitting for potentials implementing the generic linear interface.

Wraps TrainerLinearPotential with a clean Python API for
linear least-squares fitting of descriptor_linear potentials. Ordinary MTP
uses the dedicated ``gpmain trainmtp --linear-fit`` command.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Optional, List, Dict

from ._imrmlp import _LinearTrainer, _fit_nonlinear, _AnyPotential
from .atoms import Atoms


@dataclass
class TrainResult:
    """Result of potential training.

    Attributes
    ----------
    feature_count : int
        Number of features (coefficients) in the model.
    train_erms : float
        Training energy RMSE (eV/atom).
    train_frms : float
        Training force RMSE (eV/A).
    train_srms : float
        Training stress RMSE (eV).
    train_hvp_rms : float
        Unweighted RMSE of the fitted Hessian-vector observations.
    train_hvp_loss : float
        Weighted HVP loss per training structure.
    """
    feature_count: int
    train_erms: float
    train_frms: float
    train_srms: float
    train_hvp_rms: float = field(default=0.0, kw_only=True)
    train_hvp_loss: float = field(default=0.0, kw_only=True)


@dataclass
class NonlinearTrainResult(TrainResult):
    """Accepted L-BFGS point, including optional validation and Hessian metrics.

    Hessian RMS is unweighted over all matrix entries. Hessian losses are
    weighted per structure; validation metrics are None without validation data.
    """
    loss: float
    iterations: int
    valid_erms: Optional[float] = None
    valid_frms: Optional[float] = None
    valid_srms: Optional[float] = None
    train_hessian_rms: float = field(default=0.0, kw_only=True)
    train_hessian_loss: float = field(default=0.0, kw_only=True)
    valid_hessian_rms: Optional[float] = field(default=None, kw_only=True)
    valid_hessian_loss: Optional[float] = field(default=None, kw_only=True)
    valid_hvp_rms: Optional[float] = field(default=None, kw_only=True)
    valid_hvp_loss: Optional[float] = field(default=None, kw_only=True)
    test_erms: Optional[float] = field(default=None, kw_only=True)
    test_frms: Optional[float] = field(default=None, kw_only=True)
    test_stress_gpa: Optional[float] = field(default=None, kw_only=True)
    selected_l2_multiplier: Optional[float] = field(default=None, kw_only=True)
    regularization_candidates: List[Dict[str, float]] = field(default_factory=list, kw_only=True)
    hybrid_file: Optional[str] = field(default=None, kw_only=True)


class NonlinearTrainer:
    """CPU L-BFGS fitting through the shared C++ TrainNonlinear engine.

    ``trainable_groups`` freezes every coordinate outside the named groups.
    Extra settings use the same unprefixed keys as the CLI ``fit:`` section.
    Training and validation positions may be stored in Cartesian or fractional
    coordinates. Owned copies are converted to Cartesian; input structures and
    their Cartesian force, HVP and full Hessian labels remain unchanged.
    ``hessian_weight``, ``hessian_pow`` and ``hessian_fd_step`` enable a full
    Frobenius Hessian loss through central force differences. Both off-diagonal
    entries count; dedicated MTP linear assembly retains its upper-triangle loss.
    Hessian-only fitting is supported with positive weight. The returned Hessian
    RMS is unweighted over all matrix entries, and loss is per structure.
    """

    def __init__(self, *, energy_weight=1.0, force_weight=0.01,
                 stress_weight=0.0001, max_iterations=200,
                 trainable_groups=None, save_to="potential_fitted.mtp", **settings):
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._settings = {
            "energy_weight": str(energy_weight), "force_weight": str(force_weight),
            "stress_weight": str(stress_weight), "bfgs_maxiteration": str(int(max_iterations)),
            "save_to": str(save_to), **{str(k): str(v) for k, v in settings.items()},
        }
        weights = [float(self._settings.get(k, 0)) for k in
                   ("energy_weight", "force_weight", "stress_weight", "hvp_weight", "hessian_weight")]
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("EFS, HVP and full Hessian weights must be finite, nonnegative and not all zero")
        if trainable_groups is not None:
            if isinstance(trainable_groups, str):
                trainable_groups = [trainable_groups]
            if not trainable_groups:
                raise ValueError("trainable_groups must contain at least one group")
            self._settings["trainable_groups"] = " ".join(trainable_groups)

    def train(self, potential, training_set: List[Atoms],
              validation_set: Optional[List[Atoms]] = None, *,
              test_set: Optional[List[Atoms]] = None, fixed_core=None,
              hybrid_output: Optional[str] = None) -> NonlinearTrainResult:
        """Fit train data and evaluate held-out test data only at the final model.

        ``refit_train_valid=true`` fits the combined train/validation data after
        selection; validation fields then remain None. Test stress uses GPa and
        force RMSE averages Cartesian components. ``fixed_core`` fits an ordinary
        MTP to copied residual labels, with an existing MTP or ZBL core held fixed.
        ``hybrid_output`` saves the complete composite (default: save_to + '.hybrid');
        ``potential`` and the training checkpoints retain the residual MTP.
        Optional L2 search settings
        use ``validation_metric`` and ``regularization_multipliers``.
        """
        result = _fit_nonlinear(
            potential.inner, [a._impl for a in training_set],
            [] if validation_set is None else [a._impl for a in validation_set],
            self._settings,
            [] if test_set is None else [a._impl for a in test_set],
            None if fixed_core is None else fixed_core.inner,
            "" if hybrid_output is None else str(hybrid_output),
        )
        return NonlinearTrainResult(**result)


class LinearTrainer:
    """Linear potential trainer.

    Performs linear least-squares fitting by assembling the design
    matrix A^T A and solving the regularized normal equations.
    Cartesian and fractional stored positions are accepted; training uses
    Cartesian copies and preserves the input structures and their labels.

    Parameters
    ----------
    energy_weight : float
        Weight for energy in the loss function (default 1.0).
    force_weight : float
        Weight for forces (default 0.01).
    stress_weight : float
        Weight for stress (default 0.0001).
    reg_L2 : float
        L2 regularization parameter (default 1e-10).
    save_to : str
        Output filename for trained potential (default "potential_fitted").
    hvp_weight : float
        Weight of fixed-cell Hessian-vector labels (default 0, disabled).
    hvp_pow : float
        Atom-count power for HVP loss normalization (default 0).
    hvp_fd_step : float
        Maximum per-atom displacement in Angstrom for central force differences
        (default 1e-4). HVP probes and targets always use Cartesian components,
        independently of the stored position representation.

    Examples
    --------
    >>> trainer = LinearTrainer(energy_weight=1.0, force_weight=0.01)
    >>> result = trainer.train(potential, training_set)
    >>> print(f"Features: {result.feature_count}, E_RMSE: {result.train_erms:.4f}")
    """

    def __init__(self,
                 energy_weight: float = 1.0,
                 force_weight: float = 0.01,
                 stress_weight: float = 0.0001,
                 energy_pow: float = 1.0,
                 force_pow: float = 0.0,
                 stress_pow: float = 1.0,
                 reg_L2: float = 1e-10,
                 save_to: str = "potential_fitted.mtp",
                 hvp_weight: float = 0.0,
                 hvp_pow: float = 0.0,
                 hvp_fd_step: float = 1e-4,
                 ):
        self._impl = _LinearTrainer()
        self._impl.set_params({
            "energy_weight": str(energy_weight),
            "force_weight": str(force_weight),
            "stress_weight": str(stress_weight),
            "energy_pow": str(energy_pow),
            "force_pow": str(force_pow),
            "stress_pow": str(stress_pow),
            "weight_L2": str(reg_L2),
            "save_to": save_to,
            "hvp_weight": str(hvp_weight),
            "hvp_pow": str(hvp_pow),
            "hvp_fd_step": str(hvp_fd_step),
        })
        self._save_to = save_to

    def train(self, potential, training_set: List[Atoms]) -> TrainResult:
        """Train the potential on the given structures.

        Args:
            potential: imrmlp.Potential object (must support linear fitting).
            training_set: List of Atoms structures for training.

        Returns:
            TrainResult with training statistics.
        """
        pot_inner: _AnyPotential = (
            potential.inner if hasattr(potential, 'inner')
            else potential._impl.inner)

        cpp_atoms = [a._impl for a in training_set]
        self._impl.init(cpp_atoms, pot_inner)
        self._impl.train()
        self._impl.save(self._save_to)

        return TrainResult(
            feature_count=self._impl.feature_count,
            train_erms=self._impl.train_erms,
            train_frms=self._impl.train_frms,
            train_srms=self._impl.train_srms,
            train_hvp_rms=self._impl.train_hvp_rms,
            train_hvp_loss=self._impl.train_hvp_loss,
        )

    def assemble(self, potential, training_set: List[Atoms]):
        """Assemble the design matrix without solving."""
        pot_inner: _AnyPotential = (
            potential.inner if hasattr(potential, 'inner')
            else potential._impl.inner)
        cpp_atoms = [a._impl for a in training_set]
        self._impl.init(cpp_atoms, pot_inner)
        self._impl.assemble()

    def solve(self) -> TrainResult:
        """Solve after assembly. Returns training statistics."""
        self._impl.solve()
        self._impl.save(self._save_to)
        return TrainResult(
            feature_count=self._impl.feature_count,
            train_erms=self._impl.train_erms,
            train_frms=self._impl.train_frms,
            train_srms=self._impl.train_srms,
            train_hvp_rms=self._impl.train_hvp_rms,
            train_hvp_loss=self._impl.train_hvp_loss,
        )

    @property
    def feature_count(self) -> int:
        return self._impl.feature_count

    def __repr__(self) -> str:
        return f"LinearTrainer(features={self.feature_count})"
