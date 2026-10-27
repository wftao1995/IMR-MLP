# Reproducible NEB validation tools

These tools consume user-supplied potentials and structures. Recorded material
results and private datasets are not included.

- `search_paths.py`: run independently budgeted baseline and perturbed paths.
- `path_compare.py`: compare complete paths while preserving atom correspondence.

Run each script with `--help` for its input paths and available budgets. For example:

```sh
python scripts/benchmark/neb/search_paths.py --help
python scripts/validation/check_neb_saddle.py --help
```

Use the freshly built executable and matching runtime libraries, a built Python
extension (`make python`), NumPy, ASE and spglib. A path search starts from a
completed NEB run directory: it needs `neb.dat` with its PBC metadata and the
complete set of `nebimage_<i>_final.vasp` files. It also reads `neb.ini` by
default and the potential and mapping files referenced there. Variable-cell
paths additionally need `neb_reference_cell.data`. Use `--frames initial` to
read `nebimage_<i>.vasp` instead, or `--settings` to select a different settings
file. Run the example command from the repository root:

```sh
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/benchmark/neb/search_paths.py \
  --source /path/to/source --output /path/to/new-output \
  --binary "$PWD/build/release/imrmlp"
```

The tools preserve source inputs. Output manifests identify the binaries and
inputs used, while each path retains its own stopping reason and evaluation
budget. Inspect incomplete trials alongside converged candidates before comparing
barriers. See the [NEB guide](../../../docs/advanced/neb-parallel-symmetry.md).
