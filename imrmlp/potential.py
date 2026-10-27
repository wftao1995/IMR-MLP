# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Potential wrapper — load and evaluate interatomic potentials.

Supports potential types via the C++ PotentialWrapper factory:
    mtp, eamalloy, eamfs, eampoly, lj612, sw, tersoff, zbl,
    pair_table, threebody_table, shellfile, openkim, hybrid,
    hybrid_scale, mindist,
    descriptor_linear, mtpnn1, mtp_m, hmtp, mtp_les,
    acelmp (if WITH_ACELMP)
"""

from __future__ import annotations
from typing import Iterable, List, Sequence
from ._imrmlp import _PotentialWrapper, _AnyPotential
from .atoms import Atoms
from .utils import _coerce_atoms, build_type_mapping


class ShellJob:
    """An owned shell calculation; its calculator may be released after submit.

    Each result is a new Atoms object. Errors repeat on subsequent result calls.
    Releasing the last job handle waits for completion; cancellation is absent.
    """

    def __init__(self, inner):
        self._inner = inner

    def ready(self) -> bool:
        return self._inner.ready()

    def result(self) -> Atoms:
        return Atoms(_impl=self._inner.result())

    @property
    def run_directory(self) -> str:
        return self._inner.run_directory


class Potential:
    """Interatomic potential wrapper.

    Loads a potential file and provides methods for energy/force/stress
    calculation.

    Parameters
    ----------
    filename : str
        Path to the potential file.
    pottype : str
        Potential type identifier. Default: auto (legacy files use mtp).
        Common values: "mtp", "eamalloy", "eamfs", "lj612", "tersoff",
        "sw", "zbl", "openkim".

    Examples
    --------
    >>> pot = Potential("Al.mtp")
    >>> pot.calculate(atoms)
    >>> print(atoms.energy, atoms.forces)
    """

    # Known potential type constants
    MTP = "mtp"
    EAM_ALLOY = "eamalloy"
    EAM_FS = "eamfs"
    EAM_POLY = "eampoly"
    LJ612 = "lj612"
    SW = "sw"
    TERSOFF = "tersoff"
    ZBL = "zbl"
    OPENKIM = "openkim"
    HYBRID = "hybrid"
    HYBRID_SCALE = "hybrid_scale"
    PAIR_TABLE = "pair_table"
    THREEBODY_TABLE = "threebody_table"

    def __init__(self, filename: str, pottype: str = "auto"):
        self._wrapper = _PotentialWrapper(str(filename), pottype)
        self._inner: _AnyPotential = self._wrapper.inner

    @classmethod
    def from_settings(cls, settings: dict) -> "Potential":
        """Create a potential from the contents of a [pot] settings section.

        Keys omit the leading ``pot:`` prefix. For ``descriptor_linear``, use
        ``descriptor_linear:descriptor:dtype`` and the descriptor parameters.
        A nonempty ``potfile`` loads the saved model instead of a fresh template.
        """
        obj = cls.__new__(cls)
        obj._wrapper = _PotentialWrapper({str(k): str(v) for k, v in settings.items()})
        obj._inner = obj._wrapper.inner
        return obj

    @property
    def inner(self) -> _AnyPotential:
        """The underlying C++ AnyPotential object (advanced use)."""
        return self._inner

    def save(self, filename: str, profile: str = "preserve"):
        """Save an appendable MTP, or descriptor_linear model."""
        if str(filename).endswith(".gpm"):
            from ._imrmlp import _save_gpm
            return _save_gpm(self._inner, str(filename), profile)
        if profile != "preserve":
            raise ValueError("Save profiles require GPM output")
        if (self.pottype not in ("descriptor_linear")
                and not self.supports_species_append()):
            raise ValueError("Native save requires an appendable MTP or descriptor_linear")
        self._inner.save(str(filename))

    def supports_species_append(self) -> bool:
        """Whether the loaded model supports explicit species cloning."""
        return self._inner.supports_species_append()

    def append_species(self, clone_sources: Sequence[int],
                       new_elements: Sequence[str] | None = None) -> List[int]:
        """Append species cloned from existing internal species IDs.

        Existing IDs and predictions stay unchanged. Tagged models require new
        element symbols in append order. Returns the new index for each old
        parameter; cached coefficient arrays and trainers must be reinitialized.
        """
        return self._inner.append_species(list(clone_sources),
                                          [] if new_elements is None else list(new_elements))

    @property
    def coefficients(self):
        """Copy of optimizer coordinates; positive analytic parameters use logs."""
        return self._inner.coefficients

    @coefficients.setter
    def coefficients(self, values):
        self._inner.coefficients = values

    @property
    def parameter_groups(self):
        """Named half-open ranges of optimizer coordinates."""
        return self._inner.parameter_groups




    @property
    def cutoff(self) -> float:
        """Global cutoff radius (Angstrom)."""
        return self._inner.cutoff

    @cutoff.setter
    def cutoff(self, value: float):
        self._inner.cutoff = value

    @property
    def species_count(self) -> int:
        """Number of chemical species this potential supports."""
        return self._inner.species_count

    @property
    def pottype(self) -> str:
        """Potential type string."""
        return self._wrapper.pottype

    @property
    def potfile(self) -> str:
        """Potential file path."""
        return self._wrapper.potfile

    def calculate(self,
                  atoms: Atoms,
                  properties: Sequence[str] = ("energy", "forces", "stress"),
                  use_omp: bool = False):
        """Compute energy, forces, and stress in-place.

        Modifies the Atoms object: sets energy, forces, stress, and
        site_energies fields.
        Cartesian and fractional stored positions are accepted, and their
        original representation is preserved even if evaluation fails.
        Forces remain Cartesian; stress keeps the native virial convention.

        Args:
            atoms: imrmlp.Atoms object (modified in-place).
            properties: Requested properties. Energy-only requests use getE.
            use_omp: Use OpenMP EFS evaluation when available.
        """
        props = set(properties)
        original_cartesian = atoms._impl.cartesian
        original_positions = None if original_cartesian else atoms.positions.copy()
        try:
            if not original_cartesian:
                atoms.to_cartesian()
            if props <= {"energy"}:
                self._inner.getE(atoms._impl)
            elif use_omp:
                self._inner.getEFSOMP(atoms._impl)
            else:
                self._inner.getEFS(atoms._impl)
        finally:
            if not original_cartesian:
                atoms.positions = original_positions
                atoms._impl.cartesian = original_cartesian

    def hessian(self, atoms: Atoms, *, method: str = "auto", step: float = 1e-4):
        """Return an independent Cartesian atomic Hessian, shape (3*N, 3*N).

        Atom-major xyz rows/columns use eV/Angstrom^2, without mass weighting.
        The source geometry, results and labels are preserved. ``auto`` uses
        analytical curvature when available, otherwise central force differences;
        ``analytic`` requires support and ``force_fd`` always differences forces.
        For force differences, ``step`` is the Cartesian displacement in Angstrom.
        """
        return self._inner.hessian(atoms._impl, method, step)

    def submit(self, atoms: Atoms) -> ShellJob:
        """Submit a shellfile calculation with owned Cartesian input and settings.

        Scripts retain their submission-directory resource paths and receive an
        independent absolute job directory. The input object remains unchanged.
        """
        source = _coerce_atoms(atoms)
        owned = source.copy()
        if not owned._impl.cartesian:
            owned.to_cartesian()
        return ShellJob(self._inner.submit(owned._impl))

    def hessian_vector_product(self, atoms: Atoms, direction, *,
                               method: str = "auto", step: float = 1e-4):
        """Return independent Cartesian H*v, shape (3*N,), preserving atoms.

        ``direction`` is a finite one-dimensional atom-major xyz probe. No mass
        weighting is applied. ``auto`` uses analytical HVP when supported, else
        two force evaluations; ``analytic`` requires support, ``force_fd`` always
        uses forces. The largest per-atom displacement is ``step`` Angstrom;
        the resulting product retains the magnitude of the original direction.
        A zero direction gives zero. No dense Hessian is formed by this query.
        """
        return self._inner.hessian_vector_product(atoms._impl, direction, method, step)

    def supports_analytical_hessian(self) -> bool:
        """Whether this model provides an analytical atomic dense Hessian."""
        return self._inner.supports_analytical_hessian()

    def supports_analytical_hessian_vector_product(self) -> bool:
        """Whether this model provides an analytical Cartesian HVP."""
        return self._inner.supports_analytical_hessian_vector_product()

    def calculate_energy(self, atoms: Atoms) -> float:
        """Compute energy only (faster than full EFS).

        Args:
            atoms: imrmlp.Atoms object (modified in-place with energy, keeping
                its Cartesian or fractional stored coordinates).

        Returns:
            Total energy in eV.
        """
        self.calculate(atoms, properties=("energy",))
        return atoms.energy

    def calculate_many(self,
                       structures: Iterable[object],
                       properties: Sequence[str] = ("energy", "forces", "stress"),
                       use_omp: bool = False,
                       *,
                       elements: Sequence[str] | None = None) -> List[Atoms]:
        """Evaluate a batch of structures and return imrmlp Atoms copies.

        Inputs may be imrmlp.Atoms, ASE Atoms, or pymatgen Structure/Molecule.
        External ASE/pymatgen objects are converted and are not modified.

        Supply ``elements`` in the model's species order to map actual atomic
        numbers, for example ``['Ni', 'Al']``. This order stays fixed across
        frames, including frames missing an element. Elements not listed raise
        ValueError. Existing mapped imrmlp Atoms are remapped from their actual
        numbers; their source mapping and state stay unchanged.

        With ``elements=None``, existing chemmap indices are used as before.
        For raw CFG/BIN model indices without element metadata, keep that
        default; supplying ``elements`` declares that ``numbers`` are actual
        atomic numbers. No model element order is inferred automatically.
        """
        model_elements = None if elements is None else list(elements)
        type_map = None if model_elements is None else build_type_mapping(model_elements)
        out = []
        for structure in structures:
            atoms = _coerce_atoms(structure, copy_atoms=True)
            if type_map is not None:
                numbers = atoms.numbers
                try:
                    chemmap = [type_map[int(z)] for z in numbers]
                except KeyError as exc:
                    raise ValueError(f"Atomic number {exc.args[0]} is not present in elements") from None
                # Keep the converted/copied native state, including velocities,
                # charges and HVP labels, while replacing its species mapping.
                atoms = Atoms(_impl=atoms._impl, elements=model_elements)
                atoms.chemmap = chemmap
            self.calculate(atoms, properties=properties, use_omp=use_omp)
            out.append(atoms)
        return out

    def supports_linear_fit(self) -> bool:
        """Whether this potential supports direct linear fitting."""
        return self._inner.supports_linear_fit()

    def __repr__(self) -> str:
        return (f"Potential(type='{self.pottype}', "
                f"file='{self.potfile}', "
                f"cutoff={self.cutoff:.2f})")
