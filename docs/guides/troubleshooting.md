# Troubleshooting {#guide_troubleshooting}

[中文](troubleshooting_zh.md) · [Configuration](../reference/configuration.md)

## A tutorial cannot find a model, POSCAR or example directory

Check the working directory (`pwd`) and the preceding stage's success first.
`model.mtp` and `POSCAR` are placeholders for supplied inputs, not automatically
downloaded files. Without your own inputs, use the
[calculation walkthrough](../../examples/basic-calculations/README.md).
Its property directories receive an explicit copy of the cell-relaxed structure;
NEB endpoints are exports from two endpoint relaxations. Training quickstart
output defaults to `build/quickstart/`; substitute your actual `--work` throughout
later pages, including `build/first-run/` if following older examples.

For a nonempty-directory message, reuse successful results or choose a fresh
`--work`; deleting old results is unnecessary. Multi-frame POSCAR export can
append numbered suffixes. For one frame, use `--force-one --select 0`, as in the
[LAMMPS first comparison](../../LAMMPS/README.md).

## Settings are ignored or filenames begin with `=`

Use `key value`, not `key = value`, in settings files. Comments start with
`#` or `!`. A blank line ends a `[section]`; explicit `fit:key` entries belong
outside a bracket section. Duplicate keys keep their first value. Paths are
relative to the process working directory. The `.mtp` model grammar is separate
and does use equals signs.

## Training finishes without the expected optimization

Select the action explicitly: `imrmlp trainmtp train.ini --do-lbfgs` for nonlinear
L-BFGS, or `--linear-fit` for the linear coefficients. `--check-grad` and matrix
export options are diagnostics. Check the training log for accepted iterations
and the stop reason. The final snapshot is `_Trained_Potential_.mtp`;
`fit:save_to` controls trainer checkpoints. Use separate run directories.

ALGLIB's `M>N` error means the L-BFGS history is larger than the number of free
parameters: reduce `fit:bfgs_m` (the tiny quickstart uses 2).

## Missing shared library, undefined symbols or an old executable

Use an explicit executable path and inspect the library resolution:

```sh
build/release/imrmlp --version
ldd build/release/imrmlp
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use the matching build mode for the binary and shared library. After installing
to a prefix, add its `bin` to `PATH` and its `lib` to `LD_LIBRARY_PATH`.
A static project executable still requires its external system libraries.
Rebuild consumers when headers or ABI-affecting compile flags change.

## Build dependencies and memory

Install development headers, not only runtime packages. Set `MKLROOT` to the
oneMKL installation. Disabling `EIGEN_USE_MKL_ALL` alone does not remove MKL
link flags. `USE_KIM=0` disables the optional KIM interface. Use a compatible
C++23 compiler and standard library; see [build requirements](installation.md).
Reduce build parallelism if compiler processes are killed for lack of memory.

## Species mismatch or unreasonable energy/forces

Check model species order against `index.map` (one element per line). CFG
species are indices, not atomic numbers. LAMMPS types are 1-based; the indices
in these MTP `pair_style` commands are 0-based. Python/ASE needs an explicit
`elements` list in the model's order. Do not infer ordering from filenames.
Check coordinates/cell in Å, energy in eV and force in eV/Å. The native
extensive stress is in eV; ASE returns volume-normalized stress in eV/Å³.

## LAMMPS style is missing or the adapter does not compile

Use the [LAMMPS installation guide](../../LAMMPS/README.md) and rebuild LAMMPS
after installing the style. `make imrmlp` does not build LAMMPS. Keep the copied
adapter files from one source revision. Current adapters request full neighbor
lists through the public `neighbor->add_request` API; compilation errors from
writing `NeighRequest::half/full` indicate an older copied adapter.

Check `units metal`, species mapping and the chosen style's `newton`/atom-style
requirements. Compare `run 0` energy and forces against `calc_efs` before MD.
Use the [cross-style regression script](../../LAMMPS/USER-MLIP/transport/tests/run_pair_parity.sh)
when changing ordinary CPU/OpenMP/spline/transport paths. Spline and LES/SOG
models require their corresponding styles.

## Python import or ASE calculation fails

Install/build with the same Python interpreter used to run the example.
Check `python3 -c "import imrmlp; print(imrmlp.__file__)"` and inspect the
extension's shared-library dependencies. Avoid accidentally importing another
checkout from the current directory. Install ASE separately when using the
calculator. `Potential(filename)` defaults to `auto`: GPM infers its embedded type and
legacy files load as `mtp`; specify other legacy types explicitly. See the [Python guide](../reference/python.md).

## Relaxation or NEB writes results without convergence

Check the exit code and log first. `relax` returns 2 if a structure is unconverged;
NEB returns 2 for normal termination without convergence and records
`# converged` and `# status` in its summary. Last accepted outputs support
inspection and restart; they do not establish that tolerances were met. Check
mapping, distances, freedoms and endpoints before changing budgets or
initialization. Follow [relaxation](relaxation.md) and [NEB](neb.md) step by step.

## NEB interpolation overlaps atoms or finds an unexpected mechanism

Check atom identity row by row, cells and periodic displacement branches.
Equivalent atoms are not remapped automatically. Inspect initialized images,
then consider IDPP or explicit intermediate images. Changing optimizers cannot
repair incorrect atom correspondence.

## A regression requires missing data

Historical scientific regressions can need external potentials and datasets.
Read the test's fixture requirements; a missing dataset does not demonstrate
an algorithm failure. The [quickstart](../../examples/quickstart/README.md)
uses only supplied synthetic inputs and is suitable for checking a fresh build.

When reporting a problem, include the command, minimal settings, model/species
metadata, version, build mode, compiler/dependency versions and the relevant
log. Share a small reproducible input only if you have permission to distribute it.
