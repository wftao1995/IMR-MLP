# Contributing to IMR-MLP

Start with the [build guide](docs/guides/installation.md), [architecture](docs/development/architecture.md)
and [license notices](THIRD_PARTY_NOTICES.md). Keep changes focused on a scientific
or user-visible behavior. Preserve established MTP file formats, model species
ordering and the LAMMPS interface unless a documented migration is intended.

## Source layout

| Directory | Role |
| --- | --- |
| `base`, `io`, `utils` | Structure representation, file formats and settings |
| `potentials`, `basis`, `descriptors` | Model evaluation and basis functions |
| `train`, `maxvol`, `active_learn` | Fitting, selection and active learning |
| `apps` | CLI entry points |
| `python_bind`, `imrmlp` | C++ bindings and Python/ASE wrappers |
| `LAMMPS` | Separately built LAMMPS adapters; GPL notices apply |
| `unitest` | Catch2 tests and Python binding smoke tests |
| `docs`, `examples` | User guides, model notes and reproducible inputs |

Follow the nearby C++23 style. Avoid unrelated formatting changes. Include a
small regression when behavior changes; prefer synthetic inputs to private
research datasets. Keep build outputs under `build/<mode>`.

## Validation

For shared C++ changes, build and run the relevant tests and the full suite
when its fixtures are available:

```sh
make -j8 imrmlp
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
```

For a representative local C++ check use `make -j8 test-fast` with the same
`LD_LIBRARY_PATH`. The first run compiles its test objects; later shared-library
updates relink the tests without recompiling unchanged sources. The fast C++
suite does not replace the full suite. After `make python`, run
`make python-test-fast` for Python feedback without the `extended` numerical,
I/O matrix and external-runtime integration tests; `pytest` includes them.
Optional external dependencies can still cause skips in either Python suite.

Some historical tests need research data not distributed here. Set
`TEST_DATA_ROOT=/path/to/tests` and `UNITEST_DATA_ROOT=/path/to/unitest/test-data`
on the `make test` command when you have these fixtures. Report missing inputs
separately from numerical/assertion failures. To test the public workflow alone:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py --work build/quickstart-review
```

For Python changes, build with `make python`, then run `python3 -m pytest
unitest/python` from the repository root. Use the matching Python interpreter
and build library. For LAMMPS changes follow the
[cross-style and transport regressions](LAMMPS/USER-MLIP/transport/README.md)
and model-specific tests; main CLI tests do not exercise a LAMMPS pair style.

For documentation changes, update English and Chinese counterparts together,
run the examples affected by the change, and check:

```sh
make check-docs DOC_CHECK_STRICT=1
make doc-manual      # first run make doc-deps; requires Python 3.12+ and Node.js/npm
make check-doc-html
git diff --check
```

The checks compare command/option/alias names with source registrations, local
links and language pairs. They do not verify option semantics or run every example. Check `imrmlp COMMAND --help`
and the source for any changed command examples. Settings use `key value`.

Write each warning as a check, a reason and an action: name the file, column,
exit code or log field that settles the question, say briefly why it can fail,
then give the action. Do not end an introduction with a subjectless disclaimer,
and do not repeat one warning across pages; keep the full reasoning where the
mechanism is defined and link to it elsewhere. Definitional contrasts ("channels
are not levels") and safety instructions stay as written.

## Review and licensing

Describe the behavior, motivation and validation. Include build mode, required
optional dependencies and any tests that could not run. Do not commit binaries,
private datasets, machine-specific paths, credentials or generated manuals.

Use `Copyright (c) YEAR IMR-MLP contributors` and `SPDX-License-Identifier: MIT`
for new original project code. Retain upstream notices and licenses in derived
or bundled code. LAMMPS-derived adapters keep their GPLv2 notices. Do not add
third-party code without a documented source and a compatible license; update
`THIRD_PARTY_NOTICES.md` when adding dependencies. Add the complete license
notices to `config/license-files.txt` so native installations and Python wheels
include them.
