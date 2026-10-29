# IMR-MLP Python and ASE

[中文](README-python_zh.md) · [API guide](docs/reference/python.md)

Use Python 3.10+ and the native dependencies in the [build guide](docs/guides/installation.md).
Create/activate its virtual environment and, if you want the faster dense
linear algebra, enable oneMKL, then choose one installation route from the
repository root. New users can install the package and ASE:

```sh
USE_KIM=0 python3 -m pip install '.[ase]'
python3 -I -c "import imrmlp; print(imrmlp.__version__)"
```

Replace `'.[ase]'` with `.` if ASE is not needed. For in-tree use, alternatively
run `python3 -m pip install numpy pybind11` and
`make python PYTHON=python3 USE_KIM=0`; you need only one route.
Both need native development libraries. `make install` installs
the CLI/library; it does not install the Python package.

The source checkout contains an `imrmlp` package directory. When using the pip
installation, use `python3 -I` for the snippets below so that this directory
does not shadow the installed package and its extension. For the in-tree
`make python` route, use `PYTHONPATH=. python3` instead. Run either form from
the repository root so that the example's relative model paths resolve.

Complete the [CLI training quickstart](examples/quickstart/README.md) to obtain
`build/quickstart/fitted.mtp`; reuse it if installation already ran it.
Substitute the actual path if you chose another `--work`. Save the code as
`build/python-single-point.py`, then run
`python3 -I build/python-single-point.py` from the repository root (or use
`PYTHONPATH=. python3` for the in-tree build):

```python
import numpy as np
import imrmlp

# Run the CLI quickstart first, from the repository root.
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")
atoms = imrmlp.Atoms(
    numbers=[13, 13, 13],
    elements=["Al"],  # model species 0 = Al, NOT atomic number 13
    positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9.0, 7.1]],
    cell=np.eye(3) * 20.0,
    pbc=True,
)
pot.calculate(atoms)
print(atoms.energy)  # total eV
print(atoms.forces)  # eV/Å
print(atoms.stress)  # extensive stress, eV
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

`Potential(filename)` defaults to `pottype="auto"`: GPM files supply their own
model type; legacy files fall back to `mtp`. Pass the type explicitly for
legacy EAM, SW and other formats; their extensions do not select a type.
Use `Atoms(numbers=..., elements=[...])` with `elements` in model species order.
The ASE calculator also accepts this mapping. `Atoms.from_ase()` and
`Atoms.from_pymatgen()` do not accept it and retain atomic-number types.
File-based CFG input retains numerical species indices, which are not necessarily
atomic numbers.

Expect a finite total energy, a 3×3 force array, a 3×3 native stress tensor and
`build/quickstart/python-predicted.cfg`. No POSCAR is needed for this step.

With ASE installed, save the next code as `build/python-ase.py` and run
`python3 -I build/python-ase.py` from the repository root (or use
`PYTHONPATH=. python3` for the in-tree build).
Install ASE first with `python3 -m pip install ase` if needed:

```python
from ase import Atoms
from imrmlp.calculator import IMRMLPCalculator

atoms = Atoms("Al3", positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9, 7.1]],
              cell=[20, 20, 20], pbc=True)
atoms.calc = IMRMLPCalculator(
    filename="build/quickstart/fitted.mtp", pottype="mtp", elements=["Al"],
)
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Å
print(atoms.get_stress())           # eV/Å³, xx yy zz yz xz xy
```

The synthetic quickstart is an interface demonstration, not an Al material model.
Native stress is extensive (eV); ASE stress is `-atoms.stress / atoms.volume` (eV/Å³), with a sign change.
Both snippets use the same model and geometry, so energies and atomic forces
should agree; compare stress using that convention. For relaxation, NEB and
property inputs, see the [calculation walkthrough](examples/basic-calculations/README.md).
Continue to the [API guide](docs/reference/python.md) for more Python features.

For Phonopy, install the optional dependencies with
`python3 -m pip install '.[phonopy]'` and run
[09_phonopy.py](examples-python/09_phonopy.py). The optional
`imrmlp.phonopy.calculate_forces` adapter supplies ASE forces to Phonopy;
Phonopy handles force-constant fitting and phonon postprocessing.

The compiled extension is `imrmlp._imrmlp`. Original wrappers are MIT licensed;
linked dependencies retain their licenses. See [notices](THIRD_PARTY_NOTICES.md)
and [troubleshooting](docs/guides/troubleshooting.md) for loader/import issues.

`imrmlp.Dimer` is available with explicit MODECAR input, reproducible seeds and final-state results. See [saddle workflows](docs/advanced/saddle-workflows.md).
