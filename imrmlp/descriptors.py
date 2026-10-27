# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
ML Descriptors for atomic environments.

Wraps the C++ descriptor_wrapper which supports:
  - 'behler': Behler ACSF symmetry functions
  - 'bondorder': Bond Order Parameters
  - 'biso3': SO(3) bispectrum
  - 'biso4': SO(4) bispectrum

The Python wrapper exposes configuration, local or structure-averaged values,
and Cartesian position gradients for a selected center or a structure fingerprint.
The C++ 'mtm' descriptor requires a PotMTP instance and is unavailable through
this constructor.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from ._imrmlp import _Descriptor
from .atoms import Atoms


class Descriptor:
    """ML descriptor calculator.

    Parameters
    ----------
    dtype : str
        Descriptor type: 'behler', 'bondorder', 'biso3', 'biso4', 'mtm'.
    params : dict
        Type-specific parameters as keyword dict.

    Examples
    --------
    >>> desc = Descriptor("bondorder", {"bondorder:maxl": "6", "bondorder:cutoff": "5.0"})
    >>> print(desc.feature_size)
    """

    def __init__(self, dtype: str, params: Optional[Dict[str, str]] = None):
        p = params or {}
        self._impl = _Descriptor(dtype, p)

    @property
    def feature_size(self) -> int:
        """Dimension of the descriptor vector."""
        return self._impl.feature_size()

    @property
    def dtype(self) -> str:
        return self._impl.dtype

    def compute(self, atoms: Atoms) -> np.ndarray:
        """Return local values with shape ``(N, feature_size)`` in atom order.

        Cartesian and fractional stored coordinates are accepted. The input
        geometry, coordinate representation, PBC and species mapping are kept.
        The returned array owns its values independently of later computations.

        Species use ``atoms.chemmap`` without automatic remapping; construct
        Atoms with a consistent ``elements`` order when starting from atomic
        numbers.

        These are local values and do not include structure averaging. Use
        :meth:`compute_averaged` for a structure fingerprint.
        """
        return self._impl.compute(atoms._impl)

    def compute_averaged(self, atoms: Atoms) -> np.ndarray:
        """Return a structure fingerprint with shape ``(feature_size,)``.

        For SO4 ``biso4:average=inner-density``, average density coefficients
        before forming invariants. Other modes return the arithmetic mean of
        local values, matching ``calc_descriptors`` global output.

        Empty input returns a zero vector. Coordinates, PBC and species use
        the same rules as :meth:`compute`; the input stays unchanged and the
        returned array owns its values. This method returns values only.
        """
        return self._impl.compute_averaged(atoms._impl)

    def compute_averaged_with_gradient(self, atoms: Atoms
                                       ) -> tuple[np.ndarray, np.ndarray]:
        """Return structure values ``(D,)`` and their Jacobian ``(D, N, 3)``.

        Values follow :meth:`compute_averaged`. ``jac[f, i, a]`` is the
        derivative of structure feature ``f`` with respect to Cartesian
        component ``a`` of source atom ``i``, in the input atom order.
        Derivatives hold the cell, PBC and species fixed and are per Angstrom,
        including for input stored in fractional coordinates.

        For SO4 inner-density mode, differentiate the averaged density
        invariants; other modes differentiate the local feature mean. Periodic images
        are combined with their source atom and the center
        reaction is included. This differs from the selected-center Jacobian
        returned by :meth:`compute_with_gradient`.

        Empty input returns zeros with shapes ``(D,)`` and ``(D, 0, 3)``.
        The input stays unchanged and both returned arrays own their data.
        """
        return self._impl.compute_averaged_with_gradient(atoms._impl)

    def compute_averaged_with_strain_gradient(self, atoms: Atoms
                                              ) -> tuple[np.ndarray, np.ndarray]:
        """Return structure values ``(D,)`` and their strain Jacobian ``(D, 3, 3)``.

        Values follow :meth:`compute_averaged`. For raw strain ``epsilon``,
        Cartesian positions and cell rows transform as ``R @ (I + epsilon).T``.
        ``jac[f, a, b]`` differentiates structure feature ``f`` with respect
        to ``epsilon[a, b]``, keeping fractional positions, PBC and species fixed.

        SO4 inner-density mode differentiates the averaged density invariants.
        Other modes average local strain derivatives.
        Every neighbor image contributes, including periodic self images.
        All nine components use the feature's units, without shear factors
        or volume normalization. This differs from the fixed-cell position
        Jacobian, where self-image contributions cancel.

        Cartesian and fractional stored coordinates are accepted. Empty input
        returns zeros. The source is unchanged and both arrays own their data.
        """
        return self._impl.compute_averaged_with_strain_gradient(atoms._impl)

    def compute_averaged_with_cell_gradient(self, atoms: Atoms):
        """SO4 inner-density values and raw cell Jacobian ``(D,3,3)``.

        Use ``r=f@C+u`` with reference fractional ``f`` and Cartesian residual
        ``u``; the cell must be nonsingular. This is not pressure or stress.
        """
        return self._impl.compute_averaged_with_cell_gradient(atoms._impl)

    def compute_weighted_global_hvp(self, atoms: Atoms, weights, direction,
                                    include_cell: bool = False):
        """SO4 inner-density weighted value, gradient and Hessian product.

        Coordinates are atom-major Cartesian residuals, prefixed by nine
        row-major cell components when ``include_cell``. Feature weights are
        fixed. Cross-center density coupling is included.
        """
        return self._impl.compute_weighted_global_hvp(
            atoms._impl, np.asarray(weights, dtype=float),
            np.asarray(direction, dtype=float), include_cell)

    def compute_weighted_global_hessian(self, atoms: Atoms, weights,
                                        include_cell: bool = False):
        """Return SO4 inner-density weighted value, gradient and dense Hessian.

        Uses the same coordinates as :meth:`compute_weighted_global_hvp`.
        The returned arrays own their storage; this creates no global model.
        """
        return self._impl.compute_weighted_global_hessian(
            atoms._impl, np.asarray(weights, dtype=float), include_cell)

    def compute_with_gradient(self, atoms: Atoms, center: int
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local values, source atom indices and their position Jacobian.

        ``center`` is an explicit zero-based atom index. The returned arrays
        have shapes ``(D,)``, ``(K,)`` and ``(D, K, 3)``, respectively, where
        ``D = feature_size``. Indices are sorted, unique ``int64`` source atom
        indices, including the center and its neighbors. Periodic images of
        the same atom are combined, and the center derivative is included.
        Thus ``jac[f, k, a]`` differentiates local feature ``f`` with respect
        to Cartesian position component ``a`` of source atom ``indices[k]``.

        Derivatives hold the cell fixed and are per Angstrom, even for input
        stored in fractional coordinates. Geometry, coordinate representation,
        PBC and species mapping are kept, and all returned arrays own their
        values. An isolated center has one index and a zero Jacobian.

        Values remain local. This method does not differentiate the
        structure-averaged fingerprint or the cell. Invalid centers, including
        for empty input, raise IndexError.
        """
        return self._impl.compute_with_gradient(atoms._impl, center)

    def compute_with_strain_gradient(self, atoms: Atoms, center: int
                                     ) -> tuple[np.ndarray, np.ndarray]:
        """Return local values ``(D,)`` and a strain Jacobian ``(D, 3, 3)``.

        ``jac[f, a, b]`` differentiates feature ``f`` with respect to the raw
        component ``epsilon[a, b]`` of a homogeneous affine deformation.
        For column vectors, ``r_new = (I + epsilon) @ r``; row positions and
        the row-vector cell both transform as ``R_new = R @ (I + epsilon).T``.
        Fractional positions, PBC flags, species and the selected center stay
        fixed under this deformation. Cartesian axes specify ``a`` and ``b``.

        Each neighbor image contributes ``gradient[f, j, a] * displacement[j, b]``,
        including periodic self images. The derivative uses all nine raw strain
        components, with no shear factor or volume normalization. Its units
        are those of the feature, since strain is dimensionless. For a linear
        descriptor model, native virial is ``-sum_center,f w[f] * jac[f, a, b]``.

        Values remain local. Use a zero-based ``center`` as in
        :meth:`compute_with_gradient`; an invalid
        center, including empty input, raises IndexError. Cartesian and
        fractional stored coordinates are accepted. Geometry, representation,
        PBC, species and labels are preserved, and both returned arrays own
        their data. Derivatives use the current neighbor topology.
        """
        return self._impl.compute_with_strain_gradient(atoms._impl, center)
