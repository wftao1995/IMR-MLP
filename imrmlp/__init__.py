# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
imrmlp — Python interface for IMR-MLP MLIP simulation framework.

Two-layer architecture:
  - _imrmlp (pybind11 C++ module): thin bindings to libimrmlp.so
  - imrmlp (pure Python): Pythonic wrappers, ASE integration, convenience functions

Usage:
    import imrmlp

    # Read structures
    atoms = imrmlp.read_one("POSCAR")

    # Load potential and compute
    pot = imrmlp.Potential("Al.mtp")
    pot.calculate(atoms)
    print(atoms.energy)

    # ASE integration
    from imrmlp.calculator import IMRMLPCalculator
    atoms.calc = IMRMLPCalculator(filename="Al.mtp")
"""

# ── Re-export C++ bindings ──────────────────────────────────────────────
from ._imrmlp import (
    _Atoms,
    _AnyPotential,
    _PotentialWrapper,
    _Settings,
    _GPIO,
    _Relaxation,
    _NEB,
    _Dimer,
    _LinearTrainer,
    _MaxVol,
    _ActiveSelector,
    _FPSSelector,
    _ALPotential,
    _Descriptor,
    set_num_threads,
    get_num_threads,
    __version__,
    has_mpi,
)

# ── Pythonic wrappers ───────────────────────────────────────────────────
from .atoms import (
    Atoms,
    from_ase_list,
    to_ase_list,
    from_pymatgen_list,
    to_pymatgen_list,
)
from .potential import Potential, ShellJob
from .model_bundle import ModelBundle, ALSession
from .io import (read, read_one, iread, write, write_iter, write_one,
                 detect_format, open_reader, open_writer)
from .units import (unit_conversion_factor, convert_units, source_unit_system,
                    canonicalize_atoms, pressure_to_virial_factor)
from .xsf_fields import (GridLayout, ScalarField, DataGridBlock, VectorFieldView,
                         read_xsf_fields, write_xsf_fields)
from .relax import Relaxation, RelaxResult
from .neb import NEB, NEBResult
from .dimer import Dimer, DimerResult
from .train import LinearTrainer, TrainResult, NonlinearTrainer, NonlinearTrainResult
from .al import (
    MaxVol,
    ActiveSelector,
    FPSSelector,
    build_bmatrix,
    select_from_pool,
    cur_select,
    CURResult,
)
from .descriptors import Descriptor

# ── Public API ──────────────────────────────────────────────────────────
__all__ = [
    "ShellJob",
    # Core types
    "Atoms", "Potential", "ModelBundle", "ALSession",
    # I/O
    "read", "read_one", "iread", "write", "write_iter", "write_one",
    "detect_format", "open_reader", "open_writer",
    "unit_conversion_factor", "convert_units", "source_unit_system",
    "canonicalize_atoms", "pressure_to_virial_factor",
    "GridLayout", "ScalarField", "DataGridBlock", "VectorFieldView",
    "read_xsf_fields", "write_xsf_fields",
    "from_ase_list", "to_ase_list",
    "from_pymatgen_list", "to_pymatgen_list",
    # Relaxation
    "Relaxation", "RelaxResult",
    # NEB
    "NEB", "NEBResult",
    "Dimer", "DimerResult",
    # Training
    "LinearTrainer", "TrainResult", "NonlinearTrainer", "NonlinearTrainResult",
    # Active Learning
    "MaxVol", "ActiveSelector", "FPSSelector",
    "build_bmatrix", "select_from_pool",
    "cur_select", "CURResult",
    # Descriptors
    "Descriptor",
    # C++ access (advanced use)
    "_Atoms", "_AnyPotential", "_PotentialWrapper", "_Settings", "_GPIO",
    "_Relaxation", "_NEB", "_Dimer", "_LinearTrainer",
    "_MaxVol", "_ActiveSelector", "_FPSSelector", "_ALPotential",
    "_Descriptor",
    # Utils
    "set_num_threads", "get_num_threads", "__version__", "has_mpi",
]
