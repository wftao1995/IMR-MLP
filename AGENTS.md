# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++23 scientific simulation and MLIP framework with Python bindings. Core C++ modules live in top-level domain directories such as `base`, `io`, `apps`, `descriptors`, `potentials`, `train`, `maxvol`, `active_learn`, and `basis_analysis`. Command-line entry points are mainly under `apps`. Python binding code is in `python_bind`, with the Python package in `imrmlp`. C++ unit tests are in `unitest`, shared test fixtures are in `unitest/test-data`, and Python binding tests are configured under `unitest/python`. Examples and input decks are in `example-inputs`, `examples-python`, and `docs`.

## Build, Test, and Development Commands

- `make imrmlp`: build the main CLI target in `build/release` by default.
- `make all`: build `libimrmlp.so` and command targets.
- `make BUILD_MODE=debug all`: build with debug flags; valid modes include `release`, `debug`, `asan`, `relwithdebinfo`, `relwithdebinfo-asan`, `mpi`, and `mpi-debug`.
- `make test`: build `libimrmlp.so` and run all Catch2 tests from `unitest/ut_*.cpp`.
- `make python`: build the pybind11 extension module.
- `pytest`: run Python binding tests using the `pyproject.toml` pytest settings.
- `make check-syntax SYNTAX_SOURCES='path/to/source.cpp'`: run ad hoc syntax checks without emitting dependency files; use `SYNTAX_FLAGS` for additional include paths or defines.
- `make run_tests TEST_NAMES='ut_openkim_parameters'`: build and run selected Catch2 programs in an isolated temporary working directory.

When running a test executable directly, or invoking `make run_tests`, prepend the matching build directory to `LD_LIBRARY_PATH`, for example `LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH make run_tests`. Otherwise, the dynamic loader may select an older installed `libimrmlp.so` instead of the library just built in this checkout, producing stale results or ABI/symbol errors. Keep the build mode consistent (`build/debug` for debug tests, `build/asan` for ASan tests, and so on); append any required runtime library directories after the project build directory.

## Coding Style & Naming Conventions

Use `.clang-format` for C++/CUDA (clang-format 23) and `ruff.toml` for Python (Ruff 0.16.4). The format uses four spaces, a 100-column target, and expanded control statements. Preserve include order, comments, and string contents. Respect `.clang-format-ignore` and Ruff exclusions for dependencies, generated build outputs, archived code, and fixtures. Keep format-only changes separate from behavioral changes.

Use C++23-compatible code, keep includes local and minimal, and prefer existing utilities in `utils`, `base`, and nearby module directories over new helpers. Test files use the `unitest/ut_<feature>.cpp` naming pattern. Build outputs belong under `build/<mode>` and should not be committed.

## Testing Guidelines

Add or update Catch2 tests in `unitest` for C++ behavior, naming new files `ut_<area>.cpp`. Use fixtures from `unitest/test-data` when possible. For Python binding changes, add tests under `unitest/python` and run `pytest`. Run `make test` before submitting C++ changes that touch shared modules, descriptors, potentials, training, IO, or app commands.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `fix bug in maxvol` and `update python interface`; keep subjects concise and focused on the changed area. Pull requests should describe the behavioral change, list build/test commands run, mention required dependencies or environment variables
such as `MKLROOT` or `USE_MKL`, and link related issues. Include example command output or screenshots only when user-facing CLI or Python behavior changes.

## Security & Configuration Tips

Do not commit generated binaries, local build directories, temporary outputs, or private datasets. Check Makefile dependency paths before changing global build settings, especially MKL, KIM API, MPI, and vendored `deplibs` assumptions.
